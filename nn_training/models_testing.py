import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import json
import os
from datetime import datetime
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import segmentation_models as sm
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class TIFFModelTester:
    def __init__(self, models_dir, dataset_name, test_images_dir, test_masks_dir, 
                 output_dir="testing_models_results_tiff", patch_size=256, n_classes=5, overlap=64,
                 max_patches_to_save=50, min_annotation_ratio=0.30, max_annotation_ratio=0.95):

        self.models_dir = models_dir
        self.dataset_name = dataset_name
        self.test_images_dir = test_images_dir
        self.test_masks_dir = test_masks_dir
        self.output_dir = output_dir
        self.patch_size = patch_size
        self.n_classes = n_classes
        self.overlap = overlap

        self.max_patches_to_save = max_patches_to_save
        self.min_annotation_ratio = min_annotation_ratio
        self.max_annotation_ratio = max_annotation_ratio

        os.makedirs(self.output_dir, exist_ok=True)

        self.model_files = self._get_model_files()
        print(f"Found {len(self.model_files)} models for dataset '{dataset_name}'")

        self.test_image_files = sorted(glob.glob(f"{test_images_dir}/*.tif") + 
                                       glob.glob(f"{test_images_dir}/*.tiff"))
        self.test_mask_files = sorted(glob.glob(f"{test_masks_dir}/*.tif") + 
                                      glob.glob(f"{test_masks_dir}/*.tiff"))

        print(f"Found {len(self.test_image_files)} test TIFF images")
        print(f"Found {len(self.test_mask_files)} test TIFF masks")

        if len(self.test_image_files) == 0:
            print("WARNING: No TIFF images found! Check your paths.")

    def _get_model_files(self):

        all_models = glob.glob(f"{self.models_dir}/*.keras") + glob.glob(f"{self.models_dir}/*.h5")

        filtered_models = [m for m in all_models if self.dataset_name in os.path.basename(m)]
        return sorted(filtered_models)

    def load_tiff_image(self, image_path):

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def load_tiff_mask(self, mask_path):

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")

        return mask

    def extract_patches(self, image):

        if len(image.shape) == 2:
            h, w = image.shape
        else:
            h, w = image.shape[:2]

        step = self.patch_size - self.overlap

        patches = []
        positions = []

        n_patches_h = (h - self.patch_size) // step + 1
        n_patches_w = (w - self.patch_size) // step + 1

        if (n_patches_h - 1) * step + self.patch_size < h:
            n_patches_h += 1
        if (n_patches_w - 1) * step + self.patch_size < w:
            n_patches_w += 1

        print(f"  Image size: {h}x{w}, creating {n_patches_h}x{n_patches_w} = {n_patches_h * n_patches_w} patches...")

        for i in range(n_patches_h):
            for j in range(n_patches_w):
                start_h = min(i * step, h - self.patch_size)
                start_w = min(j * step, w - self.patch_size)
                end_h = start_h + self.patch_size
                end_w = start_w + self.patch_size

                if len(image.shape) == 2:
                    patch = image[start_h:end_h, start_w:end_w]
                else:
                    patch = image[start_h:end_h, start_w:end_w, :]

                patches.append(patch)
                positions.append((start_h, start_w, end_h, end_w))

        return patches, positions

    def reconstruct_from_patches(self, predictions, positions, original_shape):

        h, w = original_shape[:2]

        combined_probs = np.zeros((h, w, self.n_classes), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)

        for pred_probs, (start_h, start_w, end_h, end_w) in zip(predictions, positions):
            combined_probs[start_h:end_h, start_w:end_w] += pred_probs
            weight_map[start_h:end_h, start_w:end_w] += 1.0

        weight_map[weight_map == 0] = 1.0

        combined_probs = combined_probs / weight_map[:, :, np.newaxis]

        prediction_mask = np.argmax(combined_probs, axis=-1).astype(np.uint8)

        return prediction_mask

    def preprocess_patch(self, patch):

        patch = patch.astype(np.float32) / 255.0
        return patch

    def predict_patch(self, model, patch):

        patch_batch = np.expand_dims(patch, axis=0)

        pred = model.predict(patch_batch, verbose=0)

        return pred[0]

    def process_full_image(self, model, image_path, mask_path=None, batch_size=16, save_patches=False, patch_dir=None):

        print(f"\n  Loading image: {os.path.basename(image_path)}")
        full_image = self.load_tiff_image(image_path)
        original_shape = full_image.shape

        image_patches, positions = self.extract_patches(full_image)

        print(f"  Preprocessing {len(image_patches)} patches...")
        preprocessed_patches = [self.preprocess_patch(p) for p in image_patches]
        preprocessed_patches = np.array(preprocessed_patches)

        true_mask = None
        if mask_path and os.path.exists(mask_path):
            print(f"  Loading ground truth mask...")
            true_mask = self.load_tiff_mask(mask_path)

        print(f"  Running inference (batch_size={batch_size})...")
        patch_predictions = []
        num_patches = len(preprocessed_patches)

        for batch_start in range(0, num_patches, batch_size):
            batch_end = min(batch_start + batch_size, num_patches)
            batch = preprocessed_patches[batch_start:batch_end]

            pred_probs_batch = model.predict(batch, verbose=0)

            if len(pred_probs_batch.shape) == 4:
                for i in range(len(pred_probs_batch)):
                    patch_predictions.append(pred_probs_batch[i])
            else:
                patch_predictions.extend(pred_probs_batch)

        if save_patches and patch_dir is not None:
            os.makedirs(patch_dir, exist_ok=True)

            interesting_patches = []

            for idx, ((sh, sw, eh, ew), pred_probs) in enumerate(zip(positions, patch_predictions)):
                if true_mask is None:
                    interesting_patches.append((idx, (sh, sw, eh, ew), pred_probs))
                    continue

                patch_gt = true_mask[sh:eh, sw:ew]

                annotated_pixels = np.sum(patch_gt > 0)
                total_pixels = patch_gt.size
                annotation_ratio = annotated_pixels / total_pixels

                if self.min_annotation_ratio <= annotation_ratio <= self.max_annotation_ratio:
                    interesting_patches.append((idx, (sh, sw, eh, ew), pred_probs))

            if len(interesting_patches) > self.max_patches_to_save:
                import random
                random.seed(42)  
                sampled_patches = random.sample(interesting_patches, self.max_patches_to_save)
            else:
                sampled_patches = interesting_patches

            print(f"  Saving {len(sampled_patches)} interesting patches (from {len(patch_predictions)} total, {len(interesting_patches)} filtered)...")

            for idx, (sh, sw, eh, ew), pred_probs in sampled_patches:
                patch_img = full_image[sh:eh, sw:ew]

                if len(pred_probs.shape) == 3:
                    patch_pred_mask = np.argmax(pred_probs, axis=-1).astype(np.uint8)
                elif len(pred_probs.shape) == 2:
                    patch_pred_mask = pred_probs.astype(np.uint8)
                else:
                    print(f"  WARNING: Unexpected pred_probs shape: {pred_probs.shape}")
                    continue

                patch_gt = None
                if true_mask is not None:
                    patch_gt = true_mask[sh:eh, sw:ew]

                save_path = f"{patch_dir}/patch_{idx:04d}.png"
                self.visualize_patch(
                    patch_img=patch_img,
                    patch_gt=patch_gt,
                    patch_pred=patch_pred_mask,
                    save_path=save_path,
                    patch_index=idx
                )

        print(f"  Reconstructing full segmentation...")
        prediction_mask = self.reconstruct_from_patches(patch_predictions, positions, original_shape)

        result = {
            'prediction': prediction_mask,
            'original_shape': original_shape
        }

        if true_mask is not None:
            if true_mask.shape != prediction_mask.shape:
                print(f"  WARNING: Shape mismatch! Prediction: {prediction_mask.shape}, GT: {true_mask.shape}")
                prediction_mask = cv2.resize(
                    prediction_mask, 
                    (true_mask.shape[1], true_mask.shape[0]), 
                    interpolation=cv2.INTER_NEAREST
                )
                result['prediction'] = prediction_mask

            result['ground_truth'] = true_mask

            print(f"  Calculating metrics...")
            metrics = self.calculate_all_metrics(true_mask, prediction_mask)
            result.update(metrics)

        return result

    def calculate_all_metrics(self, y_true, y_pred):
        metrics = {}

        iou_per_class = []
        for class_id in range(self.n_classes):
            true_mask = (y_true == class_id)
            pred_mask = (y_pred == class_id)

            intersection = np.logical_and(true_mask, pred_mask).sum()
            union = np.logical_or(true_mask, pred_mask).sum()

            if union == 0:
                iou = float('nan')
            else:
                iou = intersection / union

            iou_per_class.append(iou)

        valid_ious = [iou for iou in iou_per_class if not np.isnan(iou)]
        metrics['iou_per_class'] = iou_per_class
        metrics['mean_iou'] = np.mean(valid_ious) if valid_ious else 0.0

        dice_per_class = []
        for class_id in range(self.n_classes):
            true_mask = (y_true == class_id)
            pred_mask = (y_pred == class_id)

            intersection = np.logical_and(true_mask, pred_mask).sum()
            dice = (2.0 * intersection) / (true_mask.sum() + pred_mask.sum() + 1e-7)

            dice_per_class.append(dice)

        metrics['dice_per_class'] = dice_per_class
        metrics['mean_dice'] = np.mean(dice_per_class)

        correct = (y_true == y_pred).sum()
        total = y_true.size
        metrics['pixel_accuracy'] = correct / total

        class_acc = []
        for class_id in range(self.n_classes):
            mask = (y_true == class_id)
            if mask.sum() == 0:
                class_acc.append(float('nan'))
            else:
                correct = ((y_true == class_id) & (y_pred == class_id)).sum()
                class_acc.append(correct / mask.sum())

        metrics['class_accuracy'] = class_acc

        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        precision = precision_score(y_true_flat, y_pred_flat, average=None, 
                                    labels=range(self.n_classes), zero_division=0)
        recall = recall_score(y_true_flat, y_pred_flat, average=None, 
                             labels=range(self.n_classes), zero_division=0)
        f1 = f1_score(y_true_flat, y_pred_flat, average=None, 
                     labels=range(self.n_classes), zero_division=0)

        metrics['precision_per_class'] = precision.tolist()
        metrics['recall_per_class'] = recall.tolist()
        metrics['f1_per_class'] = f1.tolist()
        metrics['mean_precision'] = np.mean(precision)
        metrics['mean_recall'] = np.mean(recall)
        metrics['mean_f1'] = np.mean(f1)

        cm = confusion_matrix(y_true_flat, y_pred_flat, labels=range(self.n_classes))
        metrics['confusion_matrix'] = cm.tolist()

        return metrics

    def test_model(self, model_path, batch_size=16):
        model_name = os.path.basename(model_path).replace('.keras', '').replace('.h5', '')
        
        print(f"Testing model: {model_name}")
        

        model_output_dir = f"{self.output_dir}/{model_name}"
        os.makedirs(model_output_dir, exist_ok=True)
        os.makedirs(f"{model_output_dir}/predictions", exist_ok=True)

        try:
            model = load_model(model_path, compile=False)
            print(f"Model loaded successfully")
            print(f"Model input shape: {model.input_shape}")
            print(f"Model output shape: {model.output_shape}")
        except Exception as e:
            print(f"Error loading model: {e}")

            try:
                custom_objects = {
                    'iou_score': sm.metrics.iou_score,
                    'f1_score': sm.metrics.f1_score,
                    'f1-score': sm.metrics.f1_score,
                }
                model = load_model(model_path, custom_objects=custom_objects, compile=False)
                print(f"Model loaded with custom objects")
            except Exception as e2:
                print(f"Error loading model with custom objects: {e2}")
                return None

        all_results = {
            'model_name': model_name,
            'model_path': model_path,
            'dataset': self.dataset_name,
            'n_classes': self.n_classes,
            'patch_size': self.patch_size,
            'overlap': self.overlap,
            'timestamp': datetime.now().isoformat(),
            'image_results': []
        }

        for img_path in self.test_image_files:
            img_name = os.path.basename(img_path)
            base_name = os.path.splitext(img_name)[0]

            mask_path = None
            for mask_file in self.test_mask_files:
                if base_name in os.path.basename(mask_file):
                    mask_path = mask_file
                    break

            print(f"\nProcessing: {img_name}")
            if mask_path:
                print(f"Ground truth: {os.path.basename(mask_path)}")
            else:
                print(f"No ground truth found")

            try:
                patch_dir = f"{model_output_dir}/patches/{base_name}"

                result = self.process_full_image(
                    model, img_path, mask_path, batch_size,
                    save_patches=True,  
                    patch_dir=patch_dir
                )

                pred_save_path = f"{model_output_dir}/predictions/{base_name}_prediction.tif"
                cv2.imwrite(pred_save_path, result['prediction'])
                print(f"  Saved prediction to: {pred_save_path}")

                self.visualize_result(result, img_path, 
                                     f"{model_output_dir}/predictions/{base_name}_visualization.png")

                image_result = {
                    'image_name': img_name,
                    'image_path': img_path,
                    'mask_path': mask_path,
                    'prediction_path': pred_save_path,
                    'original_shape': result['original_shape']
                }

                if 'mean_iou' in result:
                    image_result['metrics'] = {
                        'mean_iou': result['mean_iou'],
                        'iou_per_class': result['iou_per_class'],
                        'mean_dice': result['mean_dice'],
                        'dice_per_class': result['dice_per_class'],
                        'pixel_accuracy': result['pixel_accuracy'],
                        'class_accuracy': result['class_accuracy'],
                        'mean_precision': result['mean_precision'],
                        'precision_per_class': result['precision_per_class'],
                        'mean_recall': result['mean_recall'],
                        'recall_per_class': result['recall_per_class'],
                        'mean_f1': result['mean_f1'],
                        'f1_per_class': result['f1_per_class'],
                        'confusion_matrix': result['confusion_matrix']
                    }

                    print(f"  Mean IoU: {result['mean_iou']:.4f}")
                    print(f"  Pixel Accuracy: {result['pixel_accuracy']:.4f}")

                all_results['image_results'].append(image_result)

            except Exception as e:
                print(f"  Error processing {img_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if len(all_results['image_results']) > 0:
            metrics_list = [r['metrics'] for r in all_results['image_results'] if 'metrics' in r]

            if metrics_list:
                all_results['aggregate_metrics'] = {
                    'mean_iou': np.mean([m['mean_iou'] for m in metrics_list]),
                    'mean_dice': np.mean([m['mean_dice'] for m in metrics_list]),
                    'mean_pixel_accuracy': np.mean([m['pixel_accuracy'] for m in metrics_list]),
                    'mean_precision': np.mean([m['mean_precision'] for m in metrics_list]),
                    'mean_recall': np.mean([m['mean_recall'] for m in metrics_list]),
                    'mean_f1': np.mean([m['mean_f1'] for m in metrics_list])
                }

                
                print(f"Aggregate Metrics for {model_name}:")
                print(f"  Mean IoU: {all_results['aggregate_metrics']['mean_iou']:.4f}")
                print(f"  Mean Dice: {all_results['aggregate_metrics']['mean_dice']:.4f}")
                print(f"  Mean Pixel Accuracy: {all_results['aggregate_metrics']['mean_pixel_accuracy']:.4f}")
                

        json_path = f"{model_output_dir}/{model_name}_results.json"
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to: {json_path}")

        del model
        tf.keras.backend.clear_session()

        return all_results


    def visualize_patch(self, patch_img, patch_gt, patch_pred, save_path, patch_index):
        pred_colored = self.create_colored_mask(patch_pred)

        if patch_gt is not None:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(patch_img)
            axes[0].set_title(f'Patch {patch_index} - Oryginalny')
            axes[0].axis('off')

            gt_colored = self.create_colored_mask(patch_gt)
            axes[1].imshow(gt_colored)
            axes[1].set_title('Rzeczywiste oznaczenia (GT)')
            axes[1].axis('off')

            axes[2].imshow(pred_colored)
            axes[2].set_title('Przedykcja')
            axes[2].axis('off')
        else:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            axes[0].imshow(patch_img)
            axes[0].set_title(f'Patch {patch_index} - Original')
            axes[0].axis('off')

            axes[1].imshow(pred_colored)
            axes[1].set_title('Prediction')
            axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()

    def visualize_result(self, result, img_path, save_path):
        orig_img = self.load_tiff_image(img_path)
        img_name = os.path.basename(img_path).split('.')[0]

        max_size = 2048
        if orig_img.shape[0] > max_size or orig_img.shape[1] > max_size:
            scale = max_size / max(orig_img.shape[0], orig_img.shape[1])
            new_h = int(orig_img.shape[0] * scale)
            new_w = int(orig_img.shape[1] * scale)
            orig_img_vis = cv2.resize(orig_img, (new_w, new_h))
            pred_vis = cv2.resize(result['prediction'], (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            if 'ground_truth' in result:
                gt_vis = cv2.resize(result['ground_truth'], (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            orig_img_vis = orig_img
            pred_vis = result['prediction']
            gt_vis = result.get('ground_truth', None)

        pred_colored = self.create_colored_mask(pred_vis)

        if gt_vis is not None:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            gt_colored = self.create_colored_mask(gt_vis)

            axes[0].imshow(orig_img_vis)
            axes[0].set_title(f'Oryginalny obraz {img_name}')
            axes[0].axis('off')

            axes[1].imshow(gt_colored)
            axes[1].set_title('Rzeczywiste oznaczenia (GT)')
            axes[1].axis('off')

            axes[2].imshow(pred_colored)
            title = 'Predykcja'
            if 'mean_iou' in result:
                title += f"\nIoU: {result['mean_iou']:.3f}"
            axes[2].set_title(title)
            axes[2].axis('off')
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            axes[0].imshow(orig_img_vis)
            axes[0].set_title('Oryginalny obraz')
            axes[0].axis('off')

            axes[1].imshow(pred_colored)
            axes[1].set_title('Prediction')
            axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def create_colored_mask(self, mask):
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)

        cmap = plt.colormaps.get_cmap('tab10')

        for class_id in range(self.n_classes):
            color = (np.array(cmap(class_id)[:3]) * 255).astype(np.uint8)
            colored[mask == class_id] = color

        return colored

    def _custom_iou_metric_wrapper(self):
        def custom_iou_metric(y_true, y_pred):
            y_pred = tf.argmax(y_pred, axis=-1)
            y_true = tf.argmax(y_true, axis=-1)

            intersection = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32))
            union = tf.reduce_sum(tf.cast(y_true + y_pred, tf.float32)) - intersection

            return intersection / (union + tf.keras.backend.epsilon())

        return custom_iou_metric

    def _weighted_loss_wrapper(self):
        def weighted_categorical_crossentropy(y_true, y_pred):
            return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

        return weighted_categorical_crossentropy

    def test_all_models(self, batch_size=16):        
        print(f"Starting TIFF model testing for dataset: {self.dataset_name}")
        print(f"Found {len(self.model_files)} models to test")
        print(f"Test images: {len(self.test_image_files)}")
        print(f"Patch size: {self.patch_size}, Overlap: {self.overlap}")
        
        all_results = []

        for model_path in self.model_files:
            try:
                result = self.test_model(model_path, batch_size)
                if result is not None:
                    all_results.append(result)
            except Exception as e:
                print(f"Error testing model {model_path}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if len(all_results) > 0:
            self.generate_comparison_report(all_results)

        print(f"Testing complete! Results saved to: {self.output_dir}")
        
        return all_results

    def generate_comparison_report(self, all_results):
        comparison_dir = f"{self.output_dir}/comparison"
        os.makedirs(comparison_dir, exist_ok=True)

        model_comparison = []

        for result in all_results:
            if 'aggregate_metrics' in result:
                model_comparison.append({
                    'model_name': result['model_name'],
                    'mean_iou': result['aggregate_metrics']['mean_iou'],
                    'mean_dice': result['aggregate_metrics']['mean_dice'],
                    'mean_pixel_accuracy': result['aggregate_metrics']['mean_pixel_accuracy'],
                    'mean_precision': result['aggregate_metrics']['mean_precision'],
                    'mean_recall': result['aggregate_metrics']['mean_recall'],
                    'mean_f1': result['aggregate_metrics']['mean_f1']
                })

        if not model_comparison:
            print("No aggregate metrics available for comparison")
            return

        model_names = [m['model_name'] for m in model_comparison]
        mean_ious = [m['mean_iou'] for m in model_comparison]
        mean_dices = [m['mean_dice'] for m in model_comparison]
        mean_accs = [m['mean_pixel_accuracy'] for m in model_comparison]
        mean_f1s = [m['mean_f1'] for m in model_comparison]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        axes[0, 0].barh(model_names, mean_ious, color='steelblue')
        axes[0, 0].set_xlabel('Mean IoU')
        axes[0, 0].set_title('Mean IoU Comparison')
        axes[0, 0].grid(axis='x', alpha=0.3)
        for i, v in enumerate(mean_ious):
            axes[0, 0].text(v + 0.01, i, f'{v:.3f}', va='center')

        axes[0, 1].barh(model_names, mean_dices, color='forestgreen')
        axes[0, 1].set_xlabel('Mean Dice')
        axes[0, 1].set_title('Mean Dice Coefficient Comparison')
        axes[0, 1].grid(axis='x', alpha=0.3)
        for i, v in enumerate(mean_dices):
            axes[0, 1].text(v + 0.01, i, f'{v:.3f}', va='center')

        axes[1, 0].barh(model_names, mean_accs, color='coral')
        axes[1, 0].set_xlabel('Mean Pixel Accuracy')
        axes[1, 0].set_title('Mean Pixel Accuracy Comparison')
        axes[1, 0].grid(axis='x', alpha=0.3)
        for i, v in enumerate(mean_accs):
            axes[1, 0].text(v + 0.01, i, f'{v:.3f}', va='center')

        axes[1, 1].barh(model_names, mean_f1s, color='purple')
        axes[1, 1].set_xlabel('Mean F1-Score')
        axes[1, 1].set_title('Mean F1-Score Comparison')
        axes[1, 1].grid(axis='x', alpha=0.3)
        for i, v in enumerate(mean_f1s):
            axes[1, 1].text(v + 0.01, i, f'{v:.3f}', va='center')

        plt.tight_layout()
        plt.savefig(f"{comparison_dir}/models_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()

        with open(f"{comparison_dir}/comparison_summary.json", 'w') as f:
            json.dump(model_comparison, f, indent=2)

        best_iou_idx = np.argmax(mean_ious)

        
        print("COMPARISON SUMMARY")
        
        print(f"\nBest model by Mean IoU: {model_names[best_iou_idx]}")
        print(f"  Mean IoU: {mean_ious[best_iou_idx]:.4f}")
        print(f"  Mean Dice: {mean_dices[best_iou_idx]:.4f}")
        print(f"  Mean Pixel Accuracy: {mean_accs[best_iou_idx]:.4f}")
        print(f"  Mean F1: {mean_f1s[best_iou_idx]:.4f}")

        print("\nAll results:")
        for m in model_comparison:
            print(f"\n{m['model_name']}:")
            print(f"  Mean IoU: {m['mean_iou']:.4f}")
            print(f"  Mean Dice: {m['mean_dice']:.4f}")
            print(f"  Mean Pixel Accuracy: {m['mean_pixel_accuracy']:.4f}")
            print(f"  Mean F1: {m['mean_f1']:.4f}")
        


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Test semantic segmentation models on TIFF images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--models-dir', type=str, required=True,
                        help='Directory containing trained models')
    parser.add_argument('--test-images-dir', type=str, required=True,
                        help='Directory containing test TIFF images')

    # Optional arguments
    parser.add_argument('--test-masks-dir', type=str, default=None,
                        help='Directory containing ground truth masks (optional)')
    parser.add_argument('--output-dir', type=str, default='testing_models_results_tiff',
                        help='Output directory for results')
    parser.add_argument('--dataset-name', type=str, default='landcover.ai',
                        help='Dataset name filter for model selection')

    # Model testing parameters
    parser.add_argument('--patch-size', type=int, default=256,
                        help='Size of patches for processing')
    parser.add_argument('--overlap', type=int, default=64,
                        help='Overlap between patches')
    parser.add_argument('--n-classes', type=int, default=5,
                        help='Number of segmentation classes')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for inference')

    # Patch saving parameters
    parser.add_argument('--max-patches', type=int, default=50,
                        help='Maximum number of patches to save per image')
    parser.add_argument('--min-annotation', type=float, default=0.30,
                        help='Minimum annotation ratio for patch selection (0.0-1.0)')
    parser.add_argument('--max-annotation', type=float, default=0.95,
                        help='Maximum annotation ratio for patch selection (0.0-1.0)')

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.models_dir):
        print(f"ERROR: Models directory does not exist: {args.models_dir}")
        return
    if not os.path.exists(args.test_images_dir):
        print(f"ERROR: Test images directory does not exist: {args.test_images_dir}")
        return

    # Check for TIFF files
    print(f"Searching for TIFF files in {args.test_images_dir}...")
    tiff_files = glob.glob(f"{args.test_images_dir}/*.tif") + glob.glob(f"{args.test_images_dir}/*.tiff")
    if not tiff_files:
        print(f"ERROR: No TIFF files found in {args.test_images_dir}")
        return
    print(f"Found {len(tiff_files)} TIFF image(s)")

    # Check for masks if directory provided
    if args.test_masks_dir and os.path.exists(args.test_masks_dir):
        mask_files = glob.glob(f"{args.test_masks_dir}/*.tif") + glob.glob(f"{args.test_masks_dir}/*.tiff")
        print(f"Found {len(mask_files)} TIFF mask(s)")

    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Models directory: {args.models_dir}")
    print(f"  Dataset name filter: {args.dataset_name}")
    print(f"  Test images: {args.test_images_dir}")
    print(f"  Test masks: {args.test_masks_dir if args.test_masks_dir else 'None'}")
    print(f"  Output: {args.output_dir}")
    print(f"  Patch size: {args.patch_size}, Overlap: {args.overlap}")
    print(f"  Max patches to save: {args.max_patches} per image")
    print(f"  Annotation ratio filter: {args.min_annotation:.0%} - {args.max_annotation:.0%}")

    # Initialize tester
    tester = TIFFModelTester(
        models_dir=args.models_dir,
        dataset_name=args.dataset_name,
        test_images_dir=args.test_images_dir,
        test_masks_dir=args.test_masks_dir if args.test_masks_dir else args.test_images_dir,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        n_classes=args.n_classes,
        overlap=args.overlap,
        max_patches_to_save=args.max_patches,
        min_annotation_ratio=args.min_annotation,
        max_annotation_ratio=args.max_annotation
    )

    if len(tester.test_image_files) == 0:
        print(f"\nERROR: TIFFModelTester found no images after initialization")
        return

    # Test all models
    results = tester.test_all_models(batch_size=args.batch_size)

    print(f"\nTesting complete! Results saved to: {args.output_dir}/")
    print(f"Total models tested: {len(results)}")


if __name__ == "__main__":
    main()
