import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os
import json
from datetime import datetime
from tensorflow.keras.models import load_model
import tensorflow as tf
import segmentation_models as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import tensorflow.keras.backend as K


os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Use first GPU (change to '0,1' for multiple GPUs)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Prevents TF from allocating all GPU memory at once
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Check GPU availability
print(f"TensorFlow version: {tf.__version__}")
print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
        
        # Optional: Get GPU details
        gpu_details = tf.config.experimental.get_device_details(gpus[0])
        print(f"GPU Details: {gpu_details}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("WARNING: No GPU detected! Running on CPU.")
    print("Check your CUDA and cuDNN installation.")
    


class SegmentationInference:
    def __init__(self, model_path, images_dir, output_dir, masks_dir=None, 
                 dataset_config_path="datasets_info.json"):
        self.model_path = model_path
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.output_dir = output_dir
        self.dataset_config_path = dataset_config_path
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/predictions", exist_ok=True)
        os.makedirs(f"{self.output_dir}/overlays", exist_ok=True)
        os.makedirs(f"{self.output_dir}/class_overlays", exist_ok=True)
        
        # Initialize parameters (will be set from config)
        self.patch_size = 512
        self.overlap = 64
        self.n_classes = None
        self.backbone = 'efficientnetb3'
        self.dataset_name = 'uavid'
        
        # Load dataset configuration
        self._load_dataset_config()
        
        # Setup preprocessing
        self.preprocess_input = sm.get_preprocessing(self.backbone)
        self.scaler = MinMaxScaler()
        
        # Load model
        self.model = None
        self._load_model()
        
        # Setup color mapping for visualization
        self._setup_color_mapping()
        
        # Get ground truth masks if available
        self.mask_files = []
        if self.masks_dir and os.path.exists(self.masks_dir):
            self.mask_files = sorted(glob.glob(f"{self.masks_dir}/*.png") + 
                                    glob.glob(f"{self.masks_dir}/*.jpg"))
            print(f"Found {len(self.mask_files)} ground truth masks")
        
    def _load_dataset_config(self):
        """Load dataset configuration from JSON file"""
        try:
            with open(self.dataset_config_path, 'r') as f:
                config = json.load(f)
            
            dataset_config = config['datasets'][self.dataset_name]
            self.n_classes = dataset_config['classes']['num_classes']
            
            # Get preprocessing parameters if available
            if 'preprocessing' in dataset_config:
                preprocessing = dataset_config['preprocessing']
                if 'patch_size' in preprocessing:
                    self.patch_size = preprocessing['patch_size']
                if 'overlap' in preprocessing:
                    self.overlap = preprocessing['overlap']
            
            # Setup UAVid color mapping
            if self.dataset_name == 'uavid':
                self.class_colors_dict = dataset_config['classes']['class_colors']
                self.class_names = dataset_config['classes']['class_names']
                self.class_ids = dataset_config['classes']['class_ids']
                
                # Debug: Check color format
                print(f"Loaded {len(self.class_colors_dict)} class colors from config")
                for name, color in list(self.class_colors_dict.items())[:3]:  # Show first 3
                    print(f"  {name}: {color} (type: {type(color)}, len: {len(color) if hasattr(color, '__len__') else 'N/A'})")
                
            print(f"Loaded config for {self.dataset_name}: {self.n_classes} classes")
            print(f"Patch size: {self.patch_size}, Overlap: {self.overlap}")
            
        except FileNotFoundError:
            print(f"Warning: Config file {self.dataset_config_path} not found. Using default parameters.")
            self.n_classes = 8  # Default for UAVid
            
    def _load_model(self):
        """Load the trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Skip standalone weights files (they need architecture)
        if self.model_path.endswith('.weights.h5'):
            raise ValueError(f"Cannot load standalone .weights.h5 file without model architecture. Please use the .keras file instead.")
        
        # Fix keras.backend compatibility issues before loading model
        import keras.backend as keras_backend
        if not hasattr(keras_backend, 'sigmoid'):
            keras_backend.sigmoid = tf.nn.sigmoid
        if not hasattr(keras_backend, 'softmax'):
            keras_backend.softmax = tf.nn.softmax
        if not hasattr(keras_backend, 'relu'):
            keras_backend.relu = tf.nn.relu
        if not hasattr(keras_backend, 'swish'):
            keras_backend.swish = tf.nn.swish
        if not hasattr(keras_backend, 'hard_sigmoid'):
            keras_backend.hard_sigmoid = tf.keras.activations.hard_sigmoid
        
        # Define custom objects to handle compatibility issues
        custom_objects = {
            'binary_crossentropy_plus_jaccard_loss': sm.losses.bce_jaccard_loss,
            'iou_score': sm.metrics.iou_score,
            'f1-score': sm.metrics.f1_score,
            'f2-score': sm.metrics.f2_score,
            'precision': sm.metrics.precision,
            'recall': sm.metrics.recall,
            'sigmoid': tf.nn.sigmoid,
            'softmax': tf.nn.softmax,
            'relu': tf.nn.relu,
            'swish': tf.nn.swish,
        }
            
        try:
            if self.model_path.endswith('.keras'):
                print("Loading .keras model...")
                with tf.keras.utils.custom_object_scope(custom_objects):
                    self.model = load_model(self.model_path, custom_objects=custom_objects, compile=False)
                print("Model loaded successfully!")
                
            else:
                print("Loading full model...")
                with tf.keras.utils.custom_object_scope(custom_objects):
                    self.model = load_model(self.model_path, custom_objects=custom_objects, compile=False)
                print("Model loaded successfully!")
                
        except Exception as e:
            print(f"Primary loading method failed: {e}")
            print("Trying alternative loading method...")
            self._load_model_alternative()
                
    def _load_model_alternative(self):
        """Alternative method: recreate model and load weights manually"""
        print("Trying alternative loading method...")
        
        try:
            temp_model = load_model(self.model_path, compile=False)
            input_shape = temp_model.input_shape[1:]
            output_shape = temp_model.output_shape[-1]
            del temp_model
            
            print(f"Detected input shape: {input_shape}, output classes: {output_shape}")
            
            model_args = {
                'backbone_name': self.backbone,
                'encoder_weights': 'imagenet',
                'input_shape': input_shape,
                'classes': output_shape,
                'activation': 'softmax'
            }
            
            architectures = ['Unet', 'FPN', 'Linknet', 'PSPNet', 'DeepLabV3Plus']
            
            for arch in architectures:
                try:
                    print(f"Trying {arch} architecture...")
                    if arch == 'Unet':
                        new_model = sm.Unet(**model_args)
                    elif arch == 'FPN':
                        new_model = sm.FPN(**model_args)
                    elif arch == 'Linknet':
                        new_model = sm.Linknet(**model_args)
                    elif arch == 'PSPNet':
                        new_model = sm.PSPNet(**model_args)
                    elif arch == 'DeepLabV3Plus':
                        new_model = sm.DeepLabV3Plus(**model_args)
                    
                    original_model = load_model(self.model_path, compile=False)
                    
                    for i, layer in enumerate(new_model.layers):
                        if i < len(original_model.layers):
                            try:
                                layer.set_weights(original_model.layers[i].get_weights())
                            except:
                                pass
                    
                    self.model = new_model
                    print(f"Successfully loaded with {arch} architecture!")
                    return
                    
                except Exception as e:
                    print(f"{arch} failed: {e}")
                    continue
            
            print("All architectures failed, using original model anyway...")
            self.model = load_model(self.model_path, compile=False)
            
        except Exception as e:
            print(f"Alternative loading failed: {e}")
            raise
    
    def _setup_color_mapping(self):
        """Setup color mapping for visualization"""
        if self.dataset_name == 'uavid' and hasattr(self, 'class_colors_dict'):
            self.class_colors = {}
            for i, (class_name, class_id) in enumerate(zip(self.class_names, self.class_ids)):
                color = self.class_colors_dict[class_name]
                # Ensure RGB only (3 elements)
                if isinstance(color, (list, tuple, np.ndarray)):
                    self.class_colors[class_id] = np.array(color[:3], dtype=np.uint8).tolist()
                else:
                    self.class_colors[class_id] = [0, 0, 0]  # Default to black if invalid
        else:
            self.class_colors = {}
            for i in range(self.n_classes):
                color = plt.cm.tab10(i)[:3]  # Get RGB only
                self.class_colors[i] = [int(c * 255) for c in color]
        
        # Validate all colors are 3-element lists/arrays
        for class_id in self.class_colors.keys():
            color = self.class_colors[class_id]
            if not isinstance(color, (list, tuple, np.ndarray)) or len(color) != 3:
                print(f"Warning: Invalid color for class {class_id}: {color}, using default")
                self.class_colors[class_id] = [0, 0, 0]
        
        print(f"Color mapping setup for {len(self.class_colors)} classes")
    
    def preprocess_patch(self, patch):
        """Preprocess a single patch for model input"""
        patch = patch.astype(np.float32) / 255.0
        patch = self.preprocess_input(patch)
        patch = np.expand_dims(patch, axis=0)
        return patch
    
    def extract_patches(self, image):
        """Extract overlapping patches from image"""
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
        
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                start_h = min(i * step, h - self.patch_size)
                start_w = min(j * step, w - self.patch_size)
                end_h = start_h + self.patch_size
                end_w = start_w + self.patch_size
                
                patch = image[start_h:end_h, start_w:end_w]
                patches.append(patch)
                positions.append((start_h, start_w, end_h, end_w))
        
        return patches, positions
    
    def combine_patches(self, predictions, positions, original_shape):
        """Combine patch predictions back into full image"""
        h, w = original_shape[:2]
        
        combined_prediction = np.zeros((h, w, self.n_classes), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)
        
        for pred, (start_h, start_w, end_h, end_w) in zip(predictions, positions):
            # Fix: Ensure prediction has shape (H, W, C)
            if pred.ndim == 2:
                pred_one_hot = np.zeros((*pred.shape, self.n_classes), dtype=np.float32)
                for c in range(self.n_classes):
                    pred_one_hot[:, :, c] = (pred == c).astype(np.float32)
                pred = pred_one_hot
            elif pred.ndim == 1:
                print("ERROR: Flat prediction detected, skipping patch")
                continue
            elif pred.ndim == 3 and pred.shape[-1] != self.n_classes:
                print(f"WARNING: wrong channel count, resizing {pred.shape} → {self.n_classes}")
                pred = pred[:, :, :self.n_classes]
            
            patch_h = end_h - start_h
            patch_w = end_w - start_w
            
            if pred.shape[0] != patch_h or pred.shape[1] != patch_w:
                pred = cv2.resize(pred, (patch_w, patch_h), interpolation=cv2.INTER_LINEAR)
            
            combined_prediction[start_h:end_h, start_w:end_w] += pred
            weight_map[start_h:end_h, start_w:end_w] += 1.0
        
        weight_map[weight_map == 0] = 1
        combined_prediction /= weight_map[:, :, None]
        
        final_prediction = np.argmax(combined_prediction, axis=-1)
        
        return final_prediction, combined_prediction
    
    def calculate_all_metrics(self, y_true, y_pred):
        """Calculate all metrics like in landcover.ai code"""
        metrics = {}
        
        # IoU per class
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
        
        # Dice per class
        dice_per_class = []
        for class_id in range(self.n_classes):
            true_mask = (y_true == class_id)
            pred_mask = (y_pred == class_id)
            
            intersection = np.logical_and(true_mask, pred_mask).sum()
            dice = (2.0 * intersection) / (true_mask.sum() + pred_mask.sum() + 1e-7)
            
            dice_per_class.append(dice)
        
        metrics['dice_per_class'] = dice_per_class
        metrics['mean_dice'] = np.mean(dice_per_class)
        
        # Pixel accuracy
        correct = (y_true == y_pred).sum()
        total = y_true.size
        metrics['pixel_accuracy'] = correct / total
        
        # Class accuracy
        class_acc = []
        for class_id in range(self.n_classes):
            mask = (y_true == class_id)
            if mask.sum() == 0:
                class_acc.append(float('nan'))
            else:
                correct = ((y_true == class_id) & (y_pred == class_id)).sum()
                class_acc.append(correct / mask.sum())
        
        metrics['class_accuracy'] = class_acc
        
        # Precision, Recall, F1
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
        
        # Confusion matrix
        cm = confusion_matrix(y_true_flat, y_pred_flat, labels=range(self.n_classes))
        metrics['confusion_matrix'] = cm.tolist()
        
        return metrics
    
    def create_colored_mask(self, prediction):
        """Convert class prediction to colored mask"""
        h, w = prediction.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id, color in self.class_colors.items():
            mask = prediction == class_id
            # Ensure color is a numpy array with exactly 3 elements
            if isinstance(color, (list, tuple)):
                color = np.array(color[:3], dtype=np.uint8)
            elif isinstance(color, np.ndarray):
                color = color[:3].astype(np.uint8)
            
            # Apply color to all pixels of this class
            if np.any(mask):
                colored_mask[mask] = color
        
        return colored_mask
    
    def create_transparent_overlay(self, original_image, colored_mask, alpha=0.7):
        """Create transparent overlay of segmentation on original image"""
        original_image = original_image.astype(np.uint8)
        colored_mask = colored_mask.astype(np.uint8)
        
        has_prediction = np.any(colored_mask > 0, axis=2)
        overlay = original_image.copy()
        
        for i in range(3):
            overlay[:, :, i] = np.where(
                has_prediction,
                (1 - alpha) * original_image[:, :, i] + alpha * colored_mask[:, :, i],
                original_image[:, :, i]
            )
        
        return overlay.astype(np.uint8)
    
    def create_class_specific_overlays(self, original_image, prediction, alpha=0.7):
        """Create separate overlays for each class for detailed analysis"""
        overlays = {}
        original_image = original_image.astype(np.uint8)
        
        for class_id, color in self.class_colors.items():
            if class_id == 0:
                continue
                
            class_mask = (prediction == class_id)
            
            if np.any(class_mask):
                colored_overlay = np.zeros_like(original_image)
                colored_overlay[class_mask] = color[:3]
                
                overlay = original_image.copy()
                for i in range(3):
                    overlay[:, :, i] = np.where(
                        class_mask,
                        (1 - alpha) * original_image[:, :, i] + alpha * colored_overlay[:, :, i],
                        original_image[:, :, i]
                    )
                
                class_name = self.class_names[class_id] if hasattr(self, 'class_names') else f"Class_{class_id}"
                overlays[class_name] = overlay.astype(np.uint8)
        
        return overlays
    
    def predict_image(self, image_path, mask_path=None):
        """Run segmentation on a single image"""
        print(f"Processing: {os.path.basename(image_path)}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return None
        
        original_shape = image.shape
        print(f"Image shape: {original_shape}")
        
        # Extract patches
        patches, positions = self.extract_patches(image)
        print(f"Extracted {len(patches)} patches")
        
        # Predict on patches
        predictions = []
        for i, patch in enumerate(patches):
            if i % 10 == 0:
                print(f"Processing patch {i+1}/{len(patches)}")
            
            preprocessed_patch = self.preprocess_patch(patch)
            pred = self.model.predict(preprocessed_patch, verbose=0)[0]
            
            if pred.ndim == 2:
                pred = np.expand_dims(pred, axis=-1)
            elif pred.ndim == 1:
                print("ERROR: flat prediction shape, skipping patch")
                continue
            
            predictions.append(pred)
        
        # Combine patches
        final_prediction, confidence_map = self.combine_patches(predictions, positions, original_shape)
        
        # Create visualizations
        colored_mask = self.create_colored_mask(final_prediction)
        overlay = self.create_transparent_overlay(image, colored_mask, alpha=0.7)
        class_overlays = self.create_class_specific_overlays(image, final_prediction, alpha=0.7)
        
        result = {
            'prediction': final_prediction,
            'colored_mask': colored_mask,
            'overlay': overlay,
            'class_overlays': class_overlays,
            'confidence_map': confidence_map,
            'original_image': image,
            'original_shape': original_shape
        }
        
        # Load and calculate metrics if ground truth is available
        if mask_path and os.path.exists(mask_path):
            print(f"Loading ground truth mask: {os.path.basename(mask_path)}")
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if gt_mask is not None:
                if gt_mask.shape != final_prediction.shape:
                    print(f"WARNING: Shape mismatch! Prediction: {final_prediction.shape}, GT: {gt_mask.shape}")
                    final_prediction_resized = cv2.resize(
                        final_prediction, 
                        (gt_mask.shape[1], gt_mask.shape[0]), 
                        interpolation=cv2.INTER_NEAREST
                    )
                    result['prediction'] = final_prediction_resized
                    metrics = self.calculate_all_metrics(gt_mask, final_prediction_resized)
                else:
                    metrics = self.calculate_all_metrics(gt_mask, final_prediction)
                
                result['ground_truth'] = gt_mask
                result.update(metrics)
                
                print(f"Mean IoU: {metrics['mean_iou']:.4f}")
                print(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
        
        return result
    
    def save_results(self, results, image_name):
        """Save prediction results"""
        base_name = os.path.splitext(image_name)[0]
        
        # Save prediction mask
        pred_path = f"{self.output_dir}/predictions/{base_name}_prediction.png"
        cv2.imwrite(pred_path, results['prediction'].astype(np.uint8))
        
        # Save colored mask
        colored_path = f"{self.output_dir}/predictions/{base_name}_colored.png"
        cv2.imwrite(colored_path, results['colored_mask'])
        
        # Save main overlay
        overlay_path = f"{self.output_dir}/overlays/{base_name}_overlay.png"
        cv2.imwrite(overlay_path, results['overlay'])
        
        # Save class-specific overlays
        class_overlay_dir = f"{self.output_dir}/class_overlays/{base_name}"
        os.makedirs(class_overlay_dir, exist_ok=True)
        
        class_overlay_paths = {}
        for class_name, class_overlay in results['class_overlays'].items():
            class_overlay_path = f"{class_overlay_dir}/{class_name}_overlay.png"
            cv2.imwrite(class_overlay_path, class_overlay)
            class_overlay_paths[class_name] = class_overlay_path
        
        # Create side-by-side comparison
        combined_viz = self.create_side_by_side_comparison(results['original_image'], results['overlay'])
        combined_path = f"{self.output_dir}/overlays/{base_name}_comparison.png"
        cv2.imwrite(combined_path, combined_viz)
        
        # Visualization with ground truth if available
        if 'ground_truth' in results:
            viz_path = f"{self.output_dir}/predictions/{base_name}_visualization.png"
            self.visualize_result(results, viz_path, base_name)
        
        print(f"Results saved for {base_name}")
        
        return {
            'prediction_path': pred_path,
            'colored_path': colored_path,
            'overlay_path': overlay_path,
            'class_overlay_paths': class_overlay_paths,
            'comparison_path': combined_path
        }
    
    def create_side_by_side_comparison(self, original, overlay):
        """Create side-by-side comparison of original and overlay"""
        h, w = original.shape[:2]
        combined = np.zeros((h, w*2, 3), dtype=np.uint8)
        combined[:, :w] = original
        combined[:, w:] = overlay
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = min(h, w) / 1000
        thickness = max(1, int(font_scale * 2))
        
        cv2.putText(combined, "Original", (10, 30), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(combined, "Segmentation Overlay", (w + 10, 30), font, font_scale, (255, 255, 255), thickness)
        
        return combined
    
    def visualize_result(self, result, save_path, img_name):
        """Create visualization similar to landcover.ai code"""
        orig_img = result['original_image']
        pred = result['prediction']
        
        # Resize if too large
        max_size = 2048
        if orig_img.shape[0] > max_size or orig_img.shape[1] > max_size:
            scale = max_size / max(orig_img.shape[0], orig_img.shape[1])
            new_h = int(orig_img.shape[0] * scale)
            new_w = int(orig_img.shape[1] * scale)
            orig_img_vis = cv2.resize(orig_img, (new_w, new_h))
            pred_vis = cv2.resize(pred, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            if 'ground_truth' in result:
                gt_vis = cv2.resize(result['ground_truth'], (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            orig_img_vis = orig_img
            pred_vis = pred
            gt_vis = result.get('ground_truth', None)
        
        pred_colored = self.create_colored_mask(pred_vis)
        
        if gt_vis is not None:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            gt_colored = self.create_colored_mask(gt_vis)
            
            axes[0].imshow(cv2.cvtColor(orig_img_vis, cv2.COLOR_BGR2RGB))
            axes[0].set_title(f'Original Image {img_name}')
            axes[0].axis('off')
            
            axes[1].imshow(gt_colored)
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            
            title = 'Prediction'
            if 'mean_iou' in result:
                title += f"\nIoU: {result['mean_iou']:.3f}"
            axes[2].imshow(pred_colored)
            axes[2].set_title(title)
            axes[2].axis('off')
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            axes[0].imshow(cv2.cvtColor(orig_img_vis, cv2.COLOR_BGR2RGB))
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            axes[1].imshow(pred_colored)
            axes[1].set_title('Prediction')
            axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def process_directory(self):
        """Process all images in the input directory"""
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.JPG', '*.JPEG', '*.PNG', '*.TIF', '*.TIFF']
        image_files = []
        
        # Check if directory exists
        if not os.path.exists(self.images_dir):
            print(f"ERROR: Images directory does not exist: {self.images_dir}")
            print(f"Please check the path and try again.")
            return None
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(self.images_dir, ext)))
        
        # Also check subdirectories
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(self.images_dir, '**', ext), recursive=True))
        
        # Remove duplicates
        image_files = list(set(image_files))
        
        if not image_files:
            print(f"No images found in {self.images_dir}")
            print(f"Searched for extensions: {', '.join(image_extensions)}")
            print(f"Directory contents: {os.listdir(self.images_dir) if os.path.exists(self.images_dir) else 'Directory does not exist'}")
            return None
        
        print(f"Found {len(image_files)} images to process")
        
        # Build results structure like landcover.ai
        model_name = os.path.basename(self.model_path).replace('.keras', '').replace('.h5', '')
        
        all_results = {
            'model_name': model_name,
            'model_path': self.model_path,
            'dataset': self.dataset_name,
            'n_classes': self.n_classes,
            'patch_size': self.patch_size,
            'overlap': self.overlap,
            'timestamp': datetime.now().isoformat(),
            'image_results': []
        }
        
        # Process each image
        for i, image_path in enumerate(image_files):
            print(f"\n--- Processing image {i+1}/{len(image_files)} ---")
            
            try:
                # Find corresponding mask
                image_name = os.path.basename(image_path)
                base_name = os.path.splitext(image_name)[0]
                
                mask_path = None
                if self.mask_files:
                    for mask_file in self.mask_files:
                        if base_name in os.path.basename(mask_file):
                            mask_path = mask_file
                            break
                
                # Run prediction
                results = self.predict_image(image_path, mask_path)
                
                if results is not None:
                    # Save results
                    saved_paths = self.save_results(results, image_name)
                    
                    # Build image result structure
                    image_result = {
                        'image_name': image_name,
                        'image_path': image_path,
                        'mask_path': mask_path,
                        'prediction_path': saved_paths['prediction_path'],
                        'original_shape': results['original_shape']
                    }
                    
                    # Add metrics if available
                    if 'mean_iou' in results:
                        image_result['metrics'] = {
                            'mean_iou': results['mean_iou'],
                            'iou_per_class': results['iou_per_class'],
                            'mean_dice': results['mean_dice'],
                            'dice_per_class': results['dice_per_class'],
                            'pixel_accuracy': results['pixel_accuracy'],
                            'class_accuracy': results['class_accuracy'],
                            'mean_precision': results['mean_precision'],
                            'precision_per_class': results['precision_per_class'],
                            'mean_recall': results['mean_recall'],
                            'recall_per_class': results['recall_per_class'],
                            'mean_f1': results['mean_f1'],
                            'f1_per_class': results['f1_per_class'],
                            'confusion_matrix': results['confusion_matrix']
                        }
                    
                    all_results['image_results'].append(image_result)
                    
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Calculate aggregate metrics if available
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
                
                print(f"\nAggregate Metrics:")
                print(f"Mean IoU: {all_results['aggregate_metrics']['mean_iou']:.4f}")
                print(f"Mean Dice: {all_results['aggregate_metrics']['mean_dice']:.4f}")
                print(f"Mean Pixel Accuracy: {all_results['aggregate_metrics']['mean_pixel_accuracy']:.4f}")
                print(f"Mean F1: {all_results['aggregate_metrics']['mean_f1']:.4f}")
        
        # Save JSON results
        json_path = f"{self.output_dir}/{model_name}_results.json"
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {json_path}")
        
        # Print summary
        successful = len(all_results['image_results'])
        print(f"\nProcessing Summary:")
        print(f"Total images: {len(image_files)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(image_files) - successful}")
        print(f"Results saved to: {self.output_dir}")
        
        return all_results
    
    def visualize_sample_results(self, num_samples=3):
        """Visualize a few sample results"""
        overlay_files = glob.glob(f"{self.output_dir}/overlays/*_overlay.png")
        
        if not overlay_files:
            print("No overlay results found to visualize")
            return
        
        sample_files = overlay_files[:num_samples]
        
        fig, axes = plt.subplots(len(sample_files), 3, figsize=(15, 5*len(sample_files)))
        if len(sample_files) == 1:
            axes = axes.reshape(1, -1)
        
        for i, overlay_path in enumerate(sample_files):
            base_name = os.path.basename(overlay_path).replace('_overlay.png', '')
            
            # Load images
            overlay = cv2.imread(overlay_path)
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            
            colored_path = f"{self.output_dir}/predictions/{base_name}_colored.png"
            colored = cv2.imread(colored_path)
            colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
            
            pred_path = f"{self.output_dir}/predictions/{base_name}_prediction.png"
            prediction = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            
            # Plot
            axes[i, 0].imshow(prediction, cmap='tab10')
            axes[i, 0].set_title(f"Prediction: {base_name}")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(colored)
            axes[i, 1].set_title(f"Colored Mask: {base_name}")
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title(f"Overlay: {base_name}")
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.show()


def extract_model_parameters(model_name):
    """Extract parameters from model filename"""
    import re
    
    params = {
        'epochs': None,
        'backbone': None,
        'batch_size': None
    }
    
    # Extract epochs (e.g., "64_epochs" or "64epochs")
    epochs_match = re.search(r'(\d+)_?epochs?', model_name, re.IGNORECASE)
    if epochs_match:
        params['epochs'] = int(epochs_match.group(1))
    
    # Extract backbone (e.g., "efficientnetb3", "resnet50", "mobilenetv2")
    backbone_patterns = [
        r'(efficientnet[a-z]\d+)',
        r'(resnet\d+)',
        r'(mobilenet[a-z]?\d+)',
        r'(vgg\d+)',
        r'(densenet\d+)',
        r'(inceptionv\d+)'
    ]
    for pattern in backbone_patterns:
        backbone_match = re.search(pattern, model_name, re.IGNORECASE)
        if backbone_match:
            params['backbone'] = backbone_match.group(1).lower()
            break
    
    # Extract batch size (e.g., "batch32", "batch_32")
    batch_match = re.search(r'batch_?(\d+)', model_name, re.IGNORECASE)
    if batch_match:
        params['batch_size'] = int(batch_match.group(1))
    
    return params


def find_uavid_test_directories(base_data_dir):
    """Find UAVid test image and mask directories"""
    possible_paths = [
        # Standard structure
        (f"{base_data_dir}datasets/uavid/uavid_test/Images", 
         f"{base_data_dir}datasets/uavid/uavid_test/Labels"),
        
        # Alternative structure
        (f"{base_data_dir}datasets/uavid/test/Images",
         f"{base_data_dir}datasets/uavid/test/Labels"),
        
        # Sequence-based structure
        (f"{base_data_dir}datasets/uavid/uavid_test",
         f"{base_data_dir}datasets/uavid/uavid_test"),
        
        # Direct in uavid folder
        (f"{base_data_dir}datasets/uavid/Images",
         f"{base_data_dir}datasets/uavid/Labels"),
    ]
    
    print("\nSearching for UAVid test data...")
    
    for img_dir, mask_dir in possible_paths:
        print(f"  Checking: {img_dir}")
        
        if os.path.exists(img_dir):
            # Check for images
            img_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
                img_files.extend(glob.glob(f"{img_dir}/{ext}"))
                img_files.extend(glob.glob(f"{img_dir}/**/{ext}", recursive=True))
            
            if img_files:
                print(f"     Found {len(img_files)} images")
                
                # Check for masks
                mask_files = []
                if os.path.exists(mask_dir):
                    for ext in ['*.png', '*.jpg', '*.tif', '*.tiff']:
                        mask_files.extend(glob.glob(f"{mask_dir}/{ext}"))
                        mask_files.extend(glob.glob(f"{mask_dir}/**/{ext}", recursive=True))
                
                if mask_files:
                    print(f"     Found {len(mask_files)} masks in {mask_dir}")
                else:
                    print(f"    ⚠ No masks found in {mask_dir}")
                
                return img_dir, mask_dir if mask_files else None
    
    print("  ✗ No UAVid test images found in any standard location")
    return None, None


def get_uavid_models(models_dir, dataset_filter='uavid'):
    """Get all model files containing the dataset name"""
    all_models = glob.glob(f"{models_dir}/*.keras") + glob.glob(f"{models_dir}/*.h5")
    
    # Filter models by dataset name and exclude standalone weights files
    filtered_models = [
        m for m in all_models 
        if dataset_filter.lower() in os.path.basename(m).lower() 
        and not m.endswith('.weights.h5')  # Skip standalone weights files
    ]
    
    return sorted(filtered_models)


def generate_comparison_report(all_results, output_base_dir):
    """Generate comparison report across all models"""
    comparison_dir = f"{output_base_dir}/comparison"
    os.makedirs(comparison_dir, exist_ok=True)
    
    model_comparison = []
    
    for result in all_results:
        if 'aggregate_metrics' in result:
            model_info = {
                'model_name': result['model_name'],
                'model_path': result['model_path'],
                'mean_iou': result['aggregate_metrics']['mean_iou'],
                'mean_dice': result['aggregate_metrics']['mean_dice'],
                'mean_pixel_accuracy': result['aggregate_metrics']['mean_pixel_accuracy'],
                'mean_precision': result['aggregate_metrics']['mean_precision'],
                'mean_recall': result['aggregate_metrics']['mean_recall'],
                'mean_f1': result['aggregate_metrics']['mean_f1']
            }
            
            # Add extracted parameters
            params = extract_model_parameters(result['model_name'])
            model_info.update(params)
            
            model_comparison.append(model_info)
    
    if not model_comparison:
        print("No aggregate metrics available for comparison")
        return
    
    # Save comparison JSON
    with open(f"{comparison_dir}/comparison_summary.json", 'w') as f:
        json.dump(model_comparison, f, indent=2)
    
    # Create visualization
    model_names = [m['model_name'] for m in model_comparison]
    mean_ious = [m['mean_iou'] for m in model_comparison]
    mean_dices = [m['mean_dice'] for m in model_comparison]
    mean_accs = [m['mean_pixel_accuracy'] for m in model_comparison]
    mean_f1s = [m['mean_f1'] for m in model_comparison]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Mean IoU
    axes[0, 0].barh(range(len(model_names)), mean_ious, color='steelblue')
    axes[0, 0].set_yticks(range(len(model_names)))
    axes[0, 0].set_yticklabels(model_names, fontsize=8)
    axes[0, 0].set_xlabel('Mean IoU')
    axes[0, 0].set_title('Mean IoU Comparison')
    axes[0, 0].grid(axis='x', alpha=0.3)
    for i, v in enumerate(mean_ious):
        axes[0, 0].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=8)
    
    # Mean Dice
    axes[0, 1].barh(range(len(model_names)), mean_dices, color='forestgreen')
    axes[0, 1].set_yticks(range(len(model_names)))
    axes[0, 1].set_yticklabels(model_names, fontsize=8)
    axes[0, 1].set_xlabel('Mean Dice')
    axes[0, 1].set_title('Mean Dice Coefficient Comparison')
    axes[0, 1].grid(axis='x', alpha=0.3)
    for i, v in enumerate(mean_dices):
        axes[0, 1].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=8)
    
    # Mean Pixel Accuracy
    axes[1, 0].barh(range(len(model_names)), mean_accs, color='coral')
    axes[1, 0].set_yticks(range(len(model_names)))
    axes[1, 0].set_yticklabels(model_names, fontsize=8)
    axes[1, 0].set_xlabel('Mean Pixel Accuracy')
    axes[1, 0].set_title('Mean Pixel Accuracy Comparison')
    axes[1, 0].grid(axis='x', alpha=0.3)
    for i, v in enumerate(mean_accs):
        axes[1, 0].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=8)
    
    # Mean F1-Score
    axes[1, 1].barh(range(len(model_names)), mean_f1s, color='purple')
    axes[1, 1].set_yticks(range(len(model_names)))
    axes[1, 1].set_yticklabels(model_names, fontsize=8)
    axes[1, 1].set_xlabel('Mean F1-Score')
    axes[1, 1].set_title('Mean F1-Score Comparison')
    axes[1, 1].grid(axis='x', alpha=0.3)
    for i, v in enumerate(mean_f1s):
        axes[1, 1].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{comparison_dir}/models_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print summary
    best_iou_idx = np.argmax(mean_ious)
    
    print("COMPARISON SUMMARY")
    
    print(f"\nBest model by Mean IoU: {model_names[best_iou_idx]}")
    print(f"  Mean IoU: {mean_ious[best_iou_idx]:.4f}")
    print(f"  Mean Dice: {mean_dices[best_iou_idx]:.4f}")
    print(f"  Mean Pixel Accuracy: {mean_accs[best_iou_idx]:.4f}")
    print(f"  Mean F1: {mean_f1s[best_iou_idx]:.4f}")
    
    print("All Models Results:")
    for m in model_comparison:
        print(f"\n{m['model_name']}:")
        if m['epochs']:
            print(f"  Epochs: {m['epochs']}")
        if m['backbone']:
            print(f"  Backbone: {m['backbone']}")
        if m['batch_size']:
            print(f"  Batch Size: {m['batch_size']}")
        print(f"  Mean IoU: {m['mean_iou']:.4f}")
        print(f"  Mean Dice: {m['mean_dice']:.4f}")
        print(f"  Mean Pixel Accuracy: {m['mean_pixel_accuracy']:.4f}")
        print(f"  Mean F1: {m['mean_f1']:.4f}")
    
    print(f"Comparison report saved to: {comparison_dir}/")


# Main execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='UAVid Model Testing Suite - Test multiple models on UAVid dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--models-dir', type=str, required=True,
                        help='Directory containing trained models')
    parser.add_argument('--images-dir', type=str, required=True,
                        help='Directory containing UAVid test images')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Base output directory for results')

    # Optional arguments
    parser.add_argument('--masks-dir', type=str, default=None,
                        help='Directory containing ground truth masks (optional)')
    parser.add_argument('--config', type=str, default='datasets_info.json',
                        help='Path to datasets_info.json configuration file')
    parser.add_argument('--dataset-filter', type=str, default='uavid',
                        help='Filter models by dataset name in filename')
    parser.add_argument('--num-samples', type=int, default=3,
                        help='Number of sample results to visualize per model')

    args = parser.parse_args()

    print("=" * 60)
    print("UAVid Model Testing Suite")
    print("=" * 60)

    # Validate paths
    if not os.path.exists(args.models_dir):
        print(f"ERROR: Models directory does not exist: {args.models_dir}")
        exit(1)
    if not os.path.exists(args.images_dir):
        print(f"ERROR: Images directory does not exist: {args.images_dir}")
        exit(1)

    # Check for masks
    if args.masks_dir and not os.path.exists(args.masks_dir):
        print(f"WARNING: Masks directory does not exist: {args.masks_dir}")
        args.masks_dir = None

    print(f"\nConfiguration:")
    print(f"  Models directory: {args.models_dir}")
    print(f"  Dataset filter: {args.dataset_filter}")
    print(f"  Test images: {args.images_dir}")
    print(f"  Test masks: {args.masks_dir if args.masks_dir else 'Not provided (predictions only)'}")
    print(f"  Output base: {args.output_dir}")
    print(f"  Config file: {args.config}")

    # Get all models matching filter
    model_files = get_uavid_models(args.models_dir, args.dataset_filter)

    if not model_files:
        print(f"\nERROR: No models found matching filter '{args.dataset_filter}' in {args.models_dir}")
        exit(1)

    print(f"\nFound {len(model_files)} models to test:")
    for i, model_path in enumerate(model_files, 1):
        model_name = os.path.basename(model_path)
        params = extract_model_parameters(model_name)
        print(f"  {i}. {model_name}")
        if params['epochs']:
            print(f"     - Epochs: {params['epochs']}")
        if params['backbone']:
            print(f"     - Backbone: {params['backbone']}")
        if params['batch_size']:
            print(f"     - Batch Size: {params['batch_size']}")

    # Test each model
    all_results = []

    for i, model_path in enumerate(model_files, 1):
        model_name = os.path.basename(model_path).replace('.keras', '').replace('.h5', '')

        print(f"\n{'=' * 60}")
        print(f"Testing Model {i}/{len(model_files)}: {model_name}")
        print(f"{'=' * 60}")

        # Extract parameters from model name
        params = extract_model_parameters(model_name)
        print(f"\nExtracted parameters:")
        print(f"  Epochs: {params['epochs'] if params['epochs'] else 'Unknown'}")
        print(f"  Backbone: {params['backbone'] if params['backbone'] else 'Unknown'}")
        print(f"  Batch Size: {params['batch_size'] if params['batch_size'] else 'Unknown'}")

        # Create model-specific output directory
        model_output_dir = f"{args.output_dir}/{model_name}"

        try:
            # Initialize inference for this model
            inference = SegmentationInference(
                model_path=model_path,
                images_dir=args.images_dir,
                output_dir=model_output_dir,
                masks_dir=args.masks_dir,
                dataset_config_path=args.config
            )

            # Process all images
            print("\nStarting batch processing...")
            results = inference.process_directory()

            if results:
                all_results.append(results)

                # Visualize sample results for this model
                print(f"\nVisualizing {args.num_samples} sample results...")
                inference.visualize_sample_results(num_samples=args.num_samples)
            else:
                print(f"⚠ Warning: No results generated for {model_name}")

            # Clean up
            del inference
            tf.keras.backend.clear_session()

            print(f"\n✓ Model {model_name} testing completed successfully")

        except Exception as e:
            print(f"\n✗ Error testing model {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Generate comparison report
    if len(all_results) > 0:
        print(f"\n{'=' * 60}")
        print("Generating Comparison Report")
        print(f"{'=' * 60}")
        generate_comparison_report(all_results, args.output_dir)
    else:
        print("\n✗ No models were successfully tested")

    print(f"\n{'=' * 60}")
    print("All Testing Completed!")
    print(f"{'=' * 60}")
    print(f"Results saved to: {args.output_dir}/")
    print(f"Total models tested: {len(all_results)}/{len(model_files)}")