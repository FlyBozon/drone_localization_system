#fast code prototyping (needed to test how nn works for futher actions), written using claude ai
#TODO: dont create osobne json do każdego obrazka, tylko dla wszytskiech obrazów/modeli wspólnie, plot results, test on every model u have, create plots for thesis

import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os
import json
import random
import tensorflow as tf
from tensorflow.keras.models import load_model
import segmentation_models as sm
import os
import cv2
import numpy as np
import itertools
from matplotlib import pyplot as plt


os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU (change to '0,1' for multiple GPUs)
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
    

# Fix keras.backend compatibility issues for EfficientNet
import keras.backend as K
if not hasattr(K, 'sigmoid'):
    K.sigmoid = tf.nn.sigmoid
if not hasattr(K, 'swish'):
    K.swish = lambda x: x * tf.nn.sigmoid(x)
if not hasattr(K, 'relu'):
    K.relu = tf.nn.relu

class LandCoverTester:
    def __init__(self, model_path, test_dir, dataset_config_path="datasets_info.json", dataset_name="landcover.ai"):
        self.model_path = model_path
        self.test_images_dir = f"{test_dir}/test_images/test"
        self.test_masks_dir = f"{test_dir}/test_masks/test"
        self.dataset_name = dataset_name
        
        # Load dataset config
        self._load_dataset_config(dataset_config_path)
        
        # CRITICAL: Match training preprocessing exactly
        self.BACKBONE = 'efficientnetb0'  # Your training backbone
        self.preprocess_input = sm.get_preprocessing(self.BACKBONE)
        self.patch_size = 256  # Your training patch size
        
        # Load model
        self.model = None
        self._load_model()
        
    def _load_dataset_config(self, config_path):
        """Load dataset configuration"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        dataset = config['datasets'][self.dataset_name]
        self.n_classes = dataset['classes']['num_classes']
        
        if 'class_names' in dataset['classes']:
            self.class_names = dataset['classes']['class_names']
        else:
            self.class_names = [f'Class_{i}' for i in range(self.n_classes)]
        
        print(f"Loaded {self.dataset_name}: {self.n_classes} classes")
        print(f"Classes: {self.class_names}")
        
    def _load_model(self):
        """Load trained model with custom objects"""
        print(f"Loading model from: {self.model_path}")
        
        # Define custom objects matching your training
        custom_objects = {
            'iou_score': sm.metrics.iou_score,
            'f1-score': sm.metrics.f1_score,
        }
        
        # Add your custom weighted loss if used
        try:
            self.model = load_model(
                self.model_path, 
                custom_objects=custom_objects,
                compile=False
            )
            print("Model loaded successfully!")
            print(f"Model input shape: {self.model.input_shape}")
            print(f"Model output shape: {self.model.output_shape}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_image(self, img):
        """
        CRITICAL: Exact same preprocessing as training!
        From your training code:
        1. Normalize to 0-1 range (img / 255.0)
        2. Apply backbone preprocessing
        """
        # Step 1: Normalize to 0-1 (matching your training exactly)
        img = img.astype(np.float32) / 255.0
        
        # Step 2: Apply EfficientNet preprocessing
        img = self.preprocess_input(img)
        
        return img
    
    def preprocess_mask(self, mask):
        """Keep mask as class indices (no one-hot encoding for visualization)"""
        return mask
    
    def calculate_useful_pixels_percentage(self, mask, useful_classes=None):
        """
        Calculate percentage of useful pixels in a mask.
        Args:
            mask: numpy array with class indices
            useful_classes: list of class indices considered useful (default: all except class 0/background)
        Returns:
            float: percentage of useful pixels (0-100)
        """
        if useful_classes is None:
            # Assume class 0 is background, all others are useful
            useful_classes = list(range(1, self.n_classes))
        
        total_pixels = mask.size
        useful_pixels = np.sum(np.isin(mask, useful_classes))
        
        return (useful_pixels / total_pixels) * 100.0
    
    def extract_patches(self, image, overlap=64):
        """Extract overlapping patches from large image"""
        h, w = image.shape[:2]
        step = self.patch_size - overlap
        
        patches = []
        positions = []
        
        print(f"Image size: {h}x{w}, Patch size: {self.patch_size}x{self.patch_size}, Overlap: {overlap}")
        
        # Calculate number of patches needed
        n_patches_h = (h - self.patch_size) // step + 1
        n_patches_w = (w - self.patch_size) // step + 1
        
        # Add extra patches if needed to cover entire image
        if (n_patches_h - 1) * step + self.patch_size < h:
            n_patches_h += 1
        if (n_patches_w - 1) * step + self.patch_size < w:
            n_patches_w += 1
        
        print(f"Extracting {n_patches_h}x{n_patches_w} = {n_patches_h * n_patches_w} patches...")
        
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                # Calculate patch position
                start_h = min(i * step, h - self.patch_size)
                start_w = min(j * step, w - self.patch_size)
                end_h = start_h + self.patch_size
                end_w = start_w + self.patch_size
                
                # Extract patch
                patch = image[start_h:end_h, start_w:end_w]
                
                # Verify patch size
                if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
                    print(f"Warning: Patch {i},{j} has incorrect size: {patch.shape}")
                    continue
                
                patches.append(patch)
                positions.append((start_h, start_w, end_h, end_w))
        
        return patches, positions
    
    def combine_patches(self, predictions, positions, original_shape):
        """Combine overlapping patch predictions back into full image"""
        h, w = original_shape[:2]
        
        # Initialize accumulation arrays
        combined_probs = np.zeros((h, w, self.n_classes), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)
        
        print(f"Combining {len(predictions)} patches into {h}x{w} image...")
        
        for pred_probs, (start_h, start_w, end_h, end_w) in zip(predictions, positions):
            # Add probabilities to accumulated result
            combined_probs[start_h:end_h, start_w:end_w] += pred_probs
            weight_map[start_h:end_h, start_w:end_w] += 1.0
        
        # Avoid division by zero
        weight_map[weight_map == 0] = 1.0
        
        # Average the probabilities
        combined_probs = combined_probs / weight_map[:, :, np.newaxis]
        
        # Convert to class predictions
        prediction_mask = np.argmax(combined_probs, axis=-1)
        
        return prediction_mask, combined_probs
    
    def visualize_top_patches(self, patch_info_list, output_dir, image_name, top_n=50, useful_threshold=50.0):
        """
        Visualize and save top N patches with highest useful pixel percentage.
        
        Args:
            patch_info_list: List of dicts with keys: 'patch', 'prediction', 'useful_pct', 'position'
            output_dir: Directory to save visualizations
            image_name: Name of the source image (for naming files)
            top_n: Number of top patches to visualize
            useful_threshold: Minimum percentage of useful pixels to consider
        """
        # Filter patches by threshold
        filtered_patches = [p for p in patch_info_list if p['useful_pct'] >= useful_threshold]
        
        if len(filtered_patches) == 0:
            print(f"No patches found with >{useful_threshold}% useful pixels")
            return
        
        # Sort by useful percentage (descending)
        filtered_patches.sort(key=lambda x: x['useful_pct'], reverse=True)
        
        # Take top N
        top_patches = filtered_patches[:min(top_n, len(filtered_patches))]
        
        print(f"Visualizing top {len(top_patches)} patches (threshold: {useful_threshold}%)")
        
        # Create output directory
        top_patches_dir = os.path.join(output_dir, "top_useful_patches")
        os.makedirs(top_patches_dir, exist_ok=True)
        
        # Create grid visualization
        n_patches = len(top_patches)
        cols = 10  # 10 columns for nice grid
        rows = int(np.ceil(n_patches / cols))
        
        # Create figure for input patches
        fig_input, axes_input = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
        if rows == 1 and cols == 1:
            axes_input = np.array([axes_input])
        axes_input = axes_input.flatten()
        
        # Create figure for prediction masks
        fig_pred, axes_pred = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
        if rows == 1 and cols == 1:
            axes_pred = np.array([axes_pred])
        axes_pred = axes_pred.flatten()
        
        # Create figure for colored predictions
        fig_colored, axes_colored = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
        if rows == 1 and cols == 1:
            axes_colored = np.array([axes_colored])
        axes_colored = axes_colored.flatten()
        
        for idx, patch_info in enumerate(top_patches):
            patch = patch_info['patch']
            prediction = patch_info['prediction']
            useful_pct = patch_info['useful_pct']
            position = patch_info['position']
            
            # Input patch
            axes_input[idx].imshow(patch)
            axes_input[idx].set_title(f"{useful_pct:.1f}%", fontsize=8)
            axes_input[idx].axis('off')
            
            # Prediction mask (grayscale)
            axes_pred[idx].imshow(prediction, cmap='gray', vmin=0, vmax=self.n_classes-1)
            axes_pred[idx].set_title(f"{useful_pct:.1f}%", fontsize=8)
            axes_pred[idx].axis('off')
            
            # Colored prediction
            colored_pred = self.create_colored_mask(prediction)
            axes_colored[idx].imshow(colored_pred)
            axes_colored[idx].set_title(f"{useful_pct:.1f}%", fontsize=8)
            axes_colored[idx].axis('off')
            
            # Save individual patches with descriptive names
            patch_name = f"{image_name}_patch{idx+1:03d}_useful{useful_pct:.1f}pct_row{position[0]}_col{position[1]}"
            
            # Save input
            cv2.imwrite(
                os.path.join(top_patches_dir, f"{patch_name}_input.png"),
                cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
            )
            
            # Save grayscale mask
            cv2.imwrite(
                os.path.join(top_patches_dir, f"{patch_name}_mask.png"),
                prediction
            )
            
            # Save colored prediction
            cv2.imwrite(
                os.path.join(top_patches_dir, f"{patch_name}_colored.png"),
                cv2.cvtColor(colored_pred, cv2.COLOR_RGB2BGR)
            )
        
        # Hide unused subplots
        for idx in range(n_patches, len(axes_input)):
            axes_input[idx].axis('off')
            axes_pred[idx].axis('off')
            axes_colored[idx].axis('off')
        
        # Save grid visualizations
        fig_input.suptitle(f'{image_name} - Top {n_patches} Patches - Input Images (>{useful_threshold}% useful pixels)', 
                          fontsize=14, y=0.995)
        fig_input.tight_layout()
        fig_input.savefig(os.path.join(top_patches_dir, f"{image_name}_top_patches_input_grid.png"), 
                         dpi=150, bbox_inches='tight')
        plt.close(fig_input)
        
        fig_pred.suptitle(f'{image_name} - Top {n_patches} Patches - Predictions Grayscale (>{useful_threshold}% useful pixels)', 
                         fontsize=14, y=0.995)
        fig_pred.tight_layout()
        fig_pred.savefig(os.path.join(top_patches_dir, f"{image_name}_top_patches_prediction_grid.png"), 
                        dpi=150, bbox_inches='tight')
        plt.close(fig_pred)
        
        fig_colored.suptitle(f'{image_name} - Top {n_patches} Patches - Predictions Colored (>{useful_threshold}% useful pixels)', 
                           fontsize=14, y=0.995)
        fig_colored.tight_layout()
        fig_colored.savefig(os.path.join(top_patches_dir, f"{image_name}_top_patches_colored_grid.png"), 
                           dpi=150, bbox_inches='tight')
        plt.close(fig_colored)
        
        print(f"Saved top patches grid to: {top_patches_dir}")
        
        # Save statistics (no per-patch metrics, just useful pixel percentages)
        stats = {
            'image_name': image_name,
            'total_patches_analyzed': len(patch_info_list),
            'patches_above_threshold': len(filtered_patches),
            'top_n_visualized': len(top_patches),
            'threshold_percentage': useful_threshold,
            'top_patches_info': [
                {
                    'rank': i+1,
                    'useful_percentage': p['useful_pct'],
                    'position_row': p['position'][0],
                    'position_col': p['position'][1]
                }
                for i, p in enumerate(top_patches)
            ]
        }
        
        with open(os.path.join(top_patches_dir, f"{image_name}_patch_statistics.json"), 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Statistics: {len(filtered_patches)}/{len(patch_info_list)} patches above {useful_threshold}% threshold")
    
    def predict_single_image(self, img_path, mask_path=None, overlap=64, patch_save_dir=None, batch_size=16):
        """Predict on a single image with patchification and save each patch/mask - GPU OPTIMIZED"""
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            return None
        
        print(f"Loaded image: {img.shape}")
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_img = img_rgb.copy()
        original_shape = img_rgb.shape
        image_name = os.path.basename(img_path).split('.')[0]

        # Create folder for saving patches if provided
        if patch_save_dir:
            os.makedirs(patch_save_dir, exist_ok=True)

        # List to store patch information for useful patches visualization
        patch_info_list = []

        # Check if image needs patchification
        if img_rgb.shape[0] <= self.patch_size and img_rgb.shape[1] <= self.patch_size:
            # Process whole image as a single patch
            if img_rgb.shape[0] != self.patch_size or img_rgb.shape[1] != self.patch_size:
                img_rgb_resized = cv2.resize(img_rgb, (self.patch_size, self.patch_size))
            else:
                img_rgb_resized = img_rgb
            
            img_preprocessed = self.preprocess_image(img_rgb_resized)
            img_batch = np.expand_dims(img_preprocessed, axis=0)
            prediction_probs = self.model.predict(img_batch, verbose=0)[0]
            prediction_mask = np.argmax(prediction_probs, axis=-1)

            # Resize back to original
            if original_shape[:2] != (self.patch_size, self.patch_size):
                prediction_mask = cv2.resize(
                    prediction_mask.astype(np.uint8), 
                    (original_shape[1], original_shape[0]), 
                    interpolation=cv2.INTER_NEAREST
                )
            
            # Calculate useful pixels and store info
            useful_pct = self.calculate_useful_pixels_percentage(prediction_mask)
            patch_info_list.append({
                'patch': img_rgb,
                'prediction': prediction_mask,
                'useful_pct': useful_pct,
                'position': (0, 0, img_rgb.shape[0], img_rgb.shape[1])
            })
            
            # Save single patch if requested
            if patch_save_dir:
                patch_filename = os.path.join(patch_save_dir, f"{image_name}_row0_col0_input.png")
                #cv2.imwrite(patch_filename, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
                
                mask_filename = os.path.join(patch_save_dir, f"{image_name}_row0_col0_mask.png")
                #cv2.imwrite(mask_filename, prediction_mask)
        
        else:
            # Extract patches
            patches, positions = self.extract_patches(img_rgb, overlap=overlap)
            
            # Preprocess all patches first (CPU operation)
            print(f"Preprocessing {len(patches)} patches...")
            preprocessed_patches = []
            for i, patch in enumerate(patches):
                patch_preprocessed = self.preprocess_image(patch)
                preprocessed_patches.append(patch_preprocessed)
                
                if (i + 1) % 100 == 0:
                    print(f"Preprocessed {i+1}/{len(patches)} patches...")
            
            # Convert to numpy array for batch prediction
            preprocessed_patches = np.array(preprocessed_patches)
            
            # Batch prediction for better GPU utilization
            print(f"Running batch predictions with batch_size={batch_size}...")
            patch_predictions = []
            num_patches = len(preprocessed_patches)
            
            for batch_start in range(0, num_patches, batch_size):
                batch_end = min(batch_start + batch_size, num_patches)
                batch = preprocessed_patches[batch_start:batch_end]
                
                # Predict on batch (GPU operation)
                pred_probs_batch = self.model.predict(batch, verbose=0)
                patch_predictions.extend(pred_probs_batch)
                
                if (batch_end) % (batch_size * 5) == 0 or batch_end == num_patches:
                    print(f"Predicted {batch_end}/{num_patches} patches...")
            
            # Calculate useful pixels and store patch info
            print(f"Calculating useful pixels for {len(patches)} patches...")
            for i, (patch, pred_probs, position) in enumerate(zip(patches, patch_predictions, positions)):
                pred_mask = np.argmax(pred_probs, axis=-1)
                useful_pct = self.calculate_useful_pixels_percentage(pred_mask)
                
                patch_info_list.append({
                    'patch': patch,
                    'prediction': pred_mask,
                    'useful_pct': useful_pct,
                    'position': position
                })
                
                if (i + 1) % 100 == 0:
                    print(f"Processed {i+1}/{len(patches)} patches...")
            
            # Save patches if requested
            if patch_save_dir:
                print(f"Saving patches to {patch_save_dir}...")
                for i, (patch, pred_probs, (start_h, start_w, end_h, end_w)) in enumerate(zip(patches, patch_predictions, positions)):
                    row_idx = start_h
                    col_idx = start_w
                    
                    # Save input patch
                    patch_filename = os.path.join(patch_save_dir, f"{image_name}_row{row_idx}_col{col_idx}_input.png")
                    #cv2.imwrite(patch_filename, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
                    
                    # Save predicted mask
                    pred_mask = np.argmax(pred_probs, axis=-1)
                    mask_filename = os.path.join(patch_save_dir, f"{image_name}_row{row_idx}_col{col_idx}_mask.png")
                    #cv2.imwrite(mask_filename, pred_mask)
                    
                    if (i + 1) % 100 == 0:
                        print(f"Saved {i+1}/{len(patches)} patches...")
            
            # Combine all patches into full image prediction
            prediction_mask, prediction_probs = self.combine_patches(patch_predictions, positions, original_shape)
        
        # Visualize top useful patches
        if patch_save_dir and len(patch_info_list) > 0:
            print(f"\nVisualizing top 50 useful patches...")
            self.visualize_top_patches(
                patch_info_list=patch_info_list,
                output_dir=patch_save_dir,
                image_name=image_name,
                top_n=50,
                useful_threshold=50.0
            )
        
        # Load ground truth if available
        true_mask = None
        if mask_path and os.path.exists(mask_path):
            true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if true_mask.shape != prediction_mask.shape:
                true_mask = cv2.resize(
                    true_mask, 
                    (prediction_mask.shape[1], prediction_mask.shape[0]), 
                    interpolation=cv2.INTER_NEAREST
                )
        
        results = {
            'original': original_img,
            'prediction': prediction_mask,
            'ground_truth': true_mask
        }
        
        # Calculate metrics for the full image
        if true_mask is not None:
            full_metrics = self.calculate_all_metrics(true_mask, prediction_mask)
            results["metrics"] = full_metrics
        
        return results

    def calculate_iou(self, true_mask, pred_mask):
        """Calculate mean IoU across all classes"""
        ious = []
        for class_id in range(self.n_classes):
            true_class = (true_mask == class_id)
            pred_class = (pred_mask == class_id)
            
            intersection = np.logical_and(true_class, pred_class).sum()
            union = np.logical_or(true_class, pred_class).sum()
            
            if union == 0:
                continue
            
            iou = intersection / union
            ious.append(iou)
        
        return np.mean(ious) if ious else 0.0
    
    def calculate_pixel_accuracy(self, true_mask, pred_mask):
        """Calculate pixel-wise accuracy"""
        correct = np.sum(true_mask == pred_mask)
        total = true_mask.size
        return correct / total
    
    def create_colored_mask(self, mask):
        """Convert class indices to colored visualization"""
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Use tab10 colormap for consistency
        #cmap = plt.cm.get_cmap('tab10')
        cmap = plt.colormaps.get_cmap('tab10')

        for class_id in range(self.n_classes):
            color = (np.array(cmap(class_id)[:3]) * 255).astype(np.uint8)
            colored[mask == class_id] = color
        
        return colored
    
    def visualize_prediction(self, results, save_path=None, patch_save_dir=None):
        """Visualize prediction results"""
        n_cols = 3 if results['ground_truth'] is not None else 2
        fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
        
        if n_cols == 3:
            axes = list(axes)
        
        # Original image
        axes[0].imshow(results['original'])
        axes[0].set_title('Oryginalny obraz')
        axes[0].axis('off')

        # Ground truth (if available)
        if results['ground_truth'] is not None and n_cols == 4:
            gt_colored = self.create_colored_mask(results['ground_truth'])
            axes[1].imshow(gt_colored)
            axes[1].set_title('Rzeczywiste oznaczenia (GT)')
            axes[1].axis('off')
        
        # Prediction
        pred_colored = self.create_colored_mask(results['prediction'])

        if save_path:
            pred_path = self.get_prediction_path(save_path)
            cv2.imwrite(pred_path, cv2.cvtColor(pred_colored, cv2.COLOR_RGB2BGR))
            
            if patch_save_dir:
                # Extract image name from save_path for descriptive filename
                base_name = os.path.splitext(os.path.basename(save_path))[0]
                mask_filename = f"{base_name}_prediction_mask.png"
                mask_path = os.path.join(patch_save_dir, mask_filename)
                cv2.imwrite(mask_path, results['prediction'])
                print(f"Saved prediction mask to: {mask_path}")

        axes[2].imshow(pred_colored)
        axes[2].set_title('Predykcja')
        axes[2].axis('off')
        
        # # Overlay
        # overlay = cv2.addWeighted(results['original'], 0.6, pred_colored, 0.4, 0)
        # axes[2].imshow(overlay)
        # title = 'Overlay'
        # if 'iou' in results:
        #     title += f"\nIoU: {results['iou']:.3f}"
        # if 'pixel_accuracy' in results:
        #     title += f" | Acc: {results['pixel_accuracy']:.3f}"
        # axes[2].set_title(title)
        # axes[2].axis('off')
        
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to: {save_path}")
        
        plt.close()  # Close figure to free memory

    def print_class_distribution(self, mask, title="Class Distribution"):
        """Print class distribution statistics"""
        print(f"\n{title}:")
        unique, counts = np.unique(mask, return_counts=True)
        total = mask.size
        
        for class_id, count in zip(unique, counts):
            if class_id < len(self.class_names):
                class_name = self.class_names[class_id]
            else:
                class_name = f'Class_{class_id}'
            percentage = (count / total) * 100
            print(f"  {class_name} (ID={class_id}): {count:,} pixels ({percentage:.2f}%)")
    
    def test_specific_image(self, img_path, mask_path=None, save_path=None, overlap=64, patch_save_dir=None, batch_size=16):

        print(f"Testing image: {img_path}")
        print(f"Using overlap: {overlap} pixels")
        print(f"Batch size: {batch_size}")
        
        results = self.predict_single_image(img_path, mask_path, overlap=overlap, 
                                           patch_save_dir=patch_save_dir, batch_size=batch_size)
        
        if results is None:
            return
        
        # Print results
        if 'iou' in results:
            print(f"IoU Score: {results['iou']:.4f}")
            print(f"Pixel Accuracy: {results['pixel_accuracy']:.4f}")
        
        self.print_class_distribution(results['prediction'], "Predicted Distribution")
        if results['ground_truth'] is not None:
            self.print_class_distribution(results['ground_truth'], "Ground Truth Distribution")
        
        # Visualize
        if save_path:
            self.visualize_prediction(results, save_path, patch_save_dir)

        # Save metrics for the full processed image only (not for patches)
        if "metrics" in results and patch_save_dir:
            base = os.path.splitext(os.path.basename(img_path))[0]
            json_path = os.path.join(patch_save_dir, f"{base}_full_image_metrics.json")

            with open(json_path, "w") as f:
                json.dump(results["metrics"], f, indent=2)

            print(f"Saved full image metrics JSON: {json_path}")

        return results

    def get_prediction_path(self, save_path):
        directory, filename = os.path.split(save_path)
        name, ext = os.path.splitext(filename)

        # Default extension if none given
        if not ext:
            ext = ".png"

        model_name = os.path.splitext(os.path.basename(self.model_path))[0]
        new_filename = f"{name}__model_{model_name}{ext}"

        return os.path.join(directory, new_filename)

    def calculate_all_metrics(self, true_mask, pred_mask):
        """
        Compute a full suite of semantic-segmentation metrics.
        Returns:
            dict with:
            - confusion_matrix
            - pixel_accuracy
            - class_accuracy + mean
            - precision per class + mean
            - recall per class + mean
            - IoU per class + mean
            - Dice per class + mean
            - F1 per class + mean
        """

        n = self.n_classes
        eps = 1e-7

        # --- Confusion Matrix ---
        cm = np.zeros((n, n), dtype=np.int64)
        for t, p in zip(true_mask.flatten(), pred_mask.flatten()):
            if 0 <= t < n and 0 <= p < n:
                cm[t, p] += 1

        # True Positives, False Positives, False Negatives
        TP = np.diag(cm).astype(float)
        FP = np.sum(cm, axis=0) - TP
        FN = np.sum(cm, axis=1) - TP
        TN = np.sum(cm) - (TP + FP + FN)

        # --- Pixel Accuracy ---
        pixel_accuracy = np.sum(TP) / (np.sum(cm) + eps)

        # --- Per-Class Accuracy ---
        class_accuracy = (TP / (TP + FN + eps)).tolist()
        mean_class_acc = float(np.mean(class_accuracy))

        # --- Precision ---
        precision = (TP / (TP + FP + eps)).tolist()
        mean_precision = float(np.mean(precision))

        # --- Recall ---
        recall = (TP / (TP + FN + eps)).tolist()
        mean_recall = float(np.mean(recall))

        # --- F1 Score ---
        f1 = (2 * TP / (2 * TP + FP + FN + eps)).tolist()
        mean_f1 = float(np.mean(f1))

        # --- IoU ---
        iou = (TP / (TP + FP + FN + eps)).tolist()
        mean_iou = float(np.mean(iou))

        # --- Dice ---
        dice = (2 * TP / (2 * TP + FP + FN + eps)).tolist()
        mean_dice = float(np.mean(dice))

        return {
            "mean_iou": mean_iou,
            "iou_per_class": iou,
            "mean_dice": mean_dice,
            "dice_per_class": dice,
            "pixel_accuracy": pixel_accuracy,
            "class_accuracy": class_accuracy,
            "mean_precision": mean_precision,
            "precision_per_class": precision,
            "mean_recall": mean_recall,
            "recall_per_class": recall,
            "mean_f1": mean_f1,
            "f1_per_class": f1,
            "confusion_matrix": cm.tolist()
        }


def process(model_path, dataset_config, processing_mode="single", **kwargs):
    print(f"Model: {model_path}")
    print(f"Mode: {processing_mode}")
    
    # Initialize tester
    tester = LandCoverTester(
        model_path=model_path,
        test_dir=".",  # Dummy directory
        dataset_config_path=dataset_config,
        dataset_name="landcover.ai"
    )
    
    if processing_mode == "single":
        return _process_single(tester, **kwargs)
    elif processing_mode == "folder":
        return _process_folder(tester, **kwargs)
    elif processing_mode == "dataset":
        return _process_dataset(tester, **kwargs)
    elif processing_mode == "batch":
        return _process_batch(tester, **kwargs)
    else:
        raise ValueError(f"Unknown processing mode: {processing_mode}")


def _process_single(tester, image_path, output_dir, mask_path=None, overlap=64, batch_size=16):
    print(f"\nProcessing single image: {image_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    image_name = os.path.basename(image_path).split('.')[0]
    
    # Create output directories
    save_path = os.path.join(output_dir, image_name, image_name)
    tile_patch_dir = os.path.join(output_dir, "patches", image_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(tile_patch_dir, exist_ok=True)

    results = tester.test_specific_image(
        img_path=image_path,
        mask_path=mask_path,
        save_path=save_path,
        overlap=overlap,
        patch_save_dir=tile_patch_dir,
        batch_size=batch_size
    )
    
    print(f"Results saved to: {save_path}")
    return results


def _process_folder(tester, input_folder, output_dir, mask_folder=None, 
                    overlap=64, batch_size=16, extensions=None):
    if extensions is None:
        extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.tif', '.tiff']
    
    print(f"\nProcessing folder: {input_folder}")
    
    # Get all image files
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, f"*{ext}")))
    
    image_files = sorted(image_files)
    
    if len(image_files) == 0:
        print(f"No images found in {input_folder}")
        return []
    
    print(f"Found {len(image_files)} images")
    
    results_list = []
    
    for i, img_path in enumerate(image_files):
        print(f"\n{'='*60}")
        print(f"Processing image {i+1}/{len(image_files)}: {os.path.basename(img_path)}")
        print(f"{'='*60}")
        
        image_name = os.path.basename(img_path).split('.')[0]
        
        # Find corresponding mask if mask_folder provided
        mask_path = None
        if mask_folder:
            for ext in extensions:
                potential_mask = os.path.join(mask_folder, f"{image_name}{ext}")
                if os.path.exists(potential_mask):
                    mask_path = potential_mask
                    break
        
        # Create output directories
        save_path = os.path.join(output_dir, image_name, image_name)
        tile_patch_dir = os.path.join(output_dir, "patches", image_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        os.makedirs(tile_patch_dir, exist_ok=True)
        
        # Process image
        try:
            results = tester.test_specific_image(
                img_path=img_path,
                mask_path=mask_path,
                save_path=save_path,
                overlap=overlap,
                patch_save_dir=tile_patch_dir,
                batch_size=batch_size
            )
            
            if results is not None:
                results_list.append({
                    'image_name': image_name,
                    'image_path': img_path,
                    'results': results
                })
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    _print_summary(results_list)
    
    return results_list


def _process_dataset(tester, dataset_root, output_dir, sequences='all', 
                     process_satellite=True, process_drone=True, overlap=64, batch_size=16):
    print(f"\nProcessing UAV_VisLoc_dataset from: {dataset_root}")
    
    # Get all sequences
    if sequences == 'all':
        sequences = sorted([d for d in os.listdir(dataset_root) 
                          if os.path.isdir(os.path.join(dataset_root, d)) 
                          and d.isdigit()])
    elif isinstance(sequences, str):
        sequences = [sequences]
    
    print(f"Processing sequences: {sequences}")
    
    all_results = []
    
    for seq in sequences:
        seq_path = os.path.join(dataset_root, seq)
        if not os.path.exists(seq_path):
            print(f"Warning: Sequence {seq} not found")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing Sequence {seq}")
        print(f"{'='*60}")
        
        # Process satellite image
        if process_satellite:
            satellite_path = os.path.join(seq_path, f"satellite{seq}.tif")
            if os.path.exists(satellite_path):
                print(f"\nProcessing satellite image...")
                seq_output = os.path.join(output_dir, seq, "satellite")
                
                try:
                    results = _process_single(
                        tester, 
                        image_path=satellite_path,
                        output_dir=seq_output,
                        overlap=overlap,
                        batch_size=batch_size
                    )
                    all_results.append({
                        'sequence': seq,
                        'type': 'satellite',
                        'results': results
                    })
                except Exception as e:
                    print(f"Error processing satellite image: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Process drone images
        if process_drone:
            drone_folder = os.path.join(seq_path, "drone")
            if os.path.exists(drone_folder):
                print(f"\nProcessing drone images...")
                seq_output = os.path.join(output_dir, seq, "drone")
                
                drone_results = _process_folder(
                    tester,
                    input_folder=drone_folder,
                    output_dir=seq_output,
                    overlap=overlap,
                    batch_size=batch_size,
                    extensions=['.JPG', '.jpg']
                )
                
                for res in drone_results:
                    all_results.append({
                        'sequence': seq,
                        'type': 'drone',
                        'image_name': res['image_name'],
                        'results': res['results']
                    })
    
    print(f"Total images processed: {len(all_results)}")
    
    # Calculate statistics
    satellite_count = sum(1 for r in all_results if r['type'] == 'satellite')
    drone_count = sum(1 for r in all_results if r['type'] == 'drone')
    print(f"Satellite images: {satellite_count}")
    print(f"Drone images: {drone_count}")
    
    # Calculate mean metrics if available
    ious = [r['results']['iou'] for r in all_results 
            if r['results'] and 'iou' in r['results']]
    if ious:
        print(f"\nMean IoU: {np.mean(ious):.4f} ± {np.std(ious):.4f}")
    
    return all_results


def _process_batch(tester, image_list, output_dir, mask_list=None, overlap=64, batch_size=16):
    print(f"\nProcessing batch of {len(image_list)} images")
    
    if mask_list and len(mask_list) != len(image_list):
        raise ValueError("mask_list must have same length as image_list")
    
    results_list = []
    
    for i, img_path in enumerate(image_list):
        print(f"\n{'='*60}")
        print(f"Processing image {i+1}/{len(image_list)}: {os.path.basename(img_path)}")
        print(f"{'='*60}")
        
        mask_path = mask_list[i] if mask_list else None
        
        try:
            results = _process_single(
                tester,
                image_path=img_path,
                output_dir=output_dir,
                mask_path=mask_path,
                overlap=overlap,
                batch_size=batch_size
            )
            
            results_list.append({
                'image_path': img_path,
                'results': results
            })
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    _print_summary(results_list)
    return results_list

def _print_summary(results_list):
    print(f"Total images processed: {len(results_list)}")
    
    # Calculate statistics
    ious = []
    accs = []
    
    for item in results_list:
        results = item.get('results')
        if results and 'iou' in results:
            ious.append(results['iou'])
            accs.append(results['pixel_accuracy'])
    
    if ious:
        print(f"\nMetrics across {len(ious)} images with ground truth:")
        print(f"Mean IoU: {np.mean(ious):.4f} ± {np.std(ious):.4f}")
        print(f"Mean Pixel Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"Min IoU: {np.min(ious):.4f}")
        print(f"Max IoU: {np.max(ious):.4f}")

def process_multiple_models(
    model_paths,
    dataset_config,
    processing_mode="single",
    **kwargs
):
    """
    Runs the same image(s) through multiple models
    and produces separate output folders.
    """

    all_results = []

    for model_path in model_paths:
        print(f"PROCESSING MODEL: {model_path}")

        # Create a subfolder for results of this model
        model_name = os.path.splitext(os.path.basename(model_path))[0]

        # user may have supplied an output_dir in kwargs
        base_output_dir = kwargs.get('output_dir', "output")

        # each model gets its own subfolder
        model_output_dir = os.path.join(base_output_dir, f"model_{model_name}")
        os.makedirs(model_output_dir, exist_ok=True)

        # === IMPORTANT FIX: avoid passing output_dir twice ===
        local_kwargs = dict(kwargs)
        if "output_dir" in local_kwargs:
            del local_kwargs["output_dir"]

        # call your existing process()
        result = process(
            model_path=model_path,
            dataset_config=dataset_config,
            processing_mode=processing_mode,
            output_dir=model_output_dir,   # single clean output_dir
            **local_kwargs
        )

        all_results.append({
            "model": model_name,
            "results": result
        })

    print("ALL MODELS FINISHED")

    return all_results

def find_prediction_files(model_dir):
    prediction_files = {}
    for root, _, files in os.walk(model_dir):
        for f in files:
            fname = f.lower()
            if "__model_" in fname and fname.endswith((".png", ".jpg")):
                image_key = f.split("__model_")[0]
                prediction_files[image_key] = os.path.join(root, f)
    return prediction_files


def show_mask(mask, ax, title="", cmap="tab10"):
    ax.imshow(mask, cmap=cmap, vmin=0, vmax=mask.max())
    ax.set_title(title, fontsize=9)
    ax.axis("off")

import hashlib

def short_name(s):
    """Generate a short unique string from a long model name"""
    return hashlib.md5(s.encode()).hexdigest()[:8]

def combine_model_outputs(output_root, colormap="tab10"):
    model_dirs = [
        os.path.join(output_root, d)
        for d in os.listdir(output_root)
        if os.path.isdir(os.path.join(output_root, d)) and d.startswith("model_")
    ]
    if len(model_dirs) < 2:
        print("Need at least two model folders to combine.")
        return

    model_maps = [find_prediction_files(m) for m in model_dirs]
    model_names = [os.path.basename(m) for m in model_dirs]

    common_images = set(model_maps[0].keys())
    for mp in model_maps[1:]:
        common_images &= set(mp.keys())
    common_images = sorted(list(common_images))

    for image_key in common_images:
        masks = [cv2.imread(mp[image_key], cv2.IMREAD_GRAYSCALE) for mp in model_maps]
        masks = np.stack(masks, axis=0)
        plot_images = [(name, cv2.imread(mp[image_key], cv2.IMREAD_GRAYSCALE)) 
                       for name, mp in zip(model_names, model_maps)]

        for combo_len in range(2, len(model_dirs)+1):
            for combo in itertools.combinations(range(len(model_dirs)), combo_len):
                selected = masks[list(combo)]
                combo_ids = [short_name(model_names[i]) for i in combo]
                combo_label = "_".join(combo_ids)
                combo_title = "_".join([model_names[i] for i in combo])

                # AND
                and_mask = np.min(selected, axis=0)
                out_dir = os.path.join(output_root, f"combo_AND_{combo_label}")
                os.makedirs(out_dir, exist_ok=True)
                plt.imsave(os.path.join(out_dir, f"{image_key}__AND.png"), 
                           and_mask, cmap=colormap, vmin=0, vmax=and_mask.max())
                plot_images.append((f"AND {combo_title}", and_mask))

                # OR
                or_mask = np.max(selected, axis=0)
                out_dir = os.path.join(output_root, f"combo_OR_{combo_label}")
                os.makedirs(out_dir, exist_ok=True)
                plt.imsave(os.path.join(out_dir, f"{image_key}__OR.png"), 
                           or_mask, cmap=colormap, vmin=0, vmax=or_mask.max())
                plot_images.append((f"OR {combo_title}", or_mask))

        majority = np.round(np.mean(masks, axis=0)).astype(np.uint8)
        out_maj = os.path.join(output_root, "combo_majority_all")
        os.makedirs(out_maj, exist_ok=True)
        plt.imsave(
            os.path.join(out_maj, f"{image_key}__MAJ.png"),
            majority, cmap=colormap, vmin=0, vmax=majority.max(),
        )
        plot_images.append(("MAJORITY(all)", majority))

        cols = 3
        rows = int(np.ceil(len(plot_images) / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = axes.flatten()

        for i, (title, img) in enumerate(plot_images):
            show_mask(img, axes[i], title, cmap=colormap)

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        panel_out = os.path.join(output_root, f"{image_key}__COMPARISON_PANEL.png")
        plt.tight_layout()
        plt.savefig(panel_out, dpi=200)
        plt.close()
        print(f"Saved comparison panel: {panel_out}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Landcover.ai Model Testing - Process images with semantic segmentation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model (.keras file)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to datasets_info.json configuration file')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['single', 'folder', 'dataset', 'batch'],
                        help='Processing mode: single image, folder of images, dataset, or batch')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for results')

    # Mode-specific arguments
    parser.add_argument('--image-path', type=str, default=None,
                        help='Path to single image (for single mode)')
    parser.add_argument('--input-folder', type=str, default=None,
                        help='Path to input folder (for folder mode)')
    parser.add_argument('--mask-path', type=str, default=None,
                        help='Path to ground truth mask (for single mode)')
    parser.add_argument('--mask-folder', type=str, default=None,
                        help='Path to masks folder (for folder mode)')
    parser.add_argument('--dataset-root', type=str, default=None,
                        help='Path to dataset root (for dataset mode)')
    parser.add_argument('--image-list', type=str, nargs='+', default=None,
                        help='List of image paths (for batch mode)')
    parser.add_argument('--mask-list', type=str, nargs='+', default=None,
                        help='List of mask paths (for batch mode)')

    # Processing parameters
    parser.add_argument('--overlap', type=int, default=64,
                        help='Overlap between patches')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for inference')

    # Dataset mode options
    parser.add_argument('--sequences', type=str, default='all',
                        help='Sequences to process (for dataset mode): "all" or comma-separated list')
    parser.add_argument('--process-satellite', action='store_true', default=True,
                        help='Process satellite images (for dataset mode)')
    parser.add_argument('--process-drone', action='store_true', default=True,
                        help='Process drone images (for dataset mode)')
    parser.add_argument('--no-satellite', dest='process_satellite', action='store_false',
                        help='Skip satellite images (for dataset mode)')
    parser.add_argument('--no-drone', dest='process_drone', action='store_false',
                        help='Skip drone images (for dataset mode)')

    args = parser.parse_args()

    # Validate mode-specific requirements
    if args.mode == 'single' and not args.image_path:
        parser.error("--image-path is required for single mode")
    if args.mode == 'folder' and not args.input_folder:
        parser.error("--input-folder is required for folder mode")
    if args.mode == 'dataset' and not args.dataset_root:
        parser.error("--dataset-root is required for dataset mode")
    if args.mode == 'batch' and not args.image_list:
        parser.error("--image-list is required for batch mode")

    print(f"Landcover.ai Model Testing")
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output_dir}")

    # Prepare arguments for process function
    process_kwargs = {
        'model_path': args.model_path,
        'dataset_config': args.config,
        'processing_mode': args.mode,
        'output_dir': args.output_dir,
        'overlap': args.overlap,
        'batch_size': args.batch_size
    }

    # Add mode-specific arguments
    if args.mode == 'single':
        process_kwargs['image_path'] = args.image_path
        if args.mask_path:
            process_kwargs['mask_path'] = args.mask_path

    elif args.mode == 'folder':
        process_kwargs['input_folder'] = args.input_folder
        if args.mask_folder:
            process_kwargs['mask_folder'] = args.mask_folder

    elif args.mode == 'dataset':
        process_kwargs['dataset_root'] = args.dataset_root
        process_kwargs['process_satellite'] = args.process_satellite
        process_kwargs['process_drone'] = args.process_drone
        if args.sequences != 'all':
            process_kwargs['sequences'] = args.sequences.split(',')

    elif args.mode == 'batch':
        process_kwargs['image_list'] = args.image_list
        if args.mask_list:
            process_kwargs['mask_list'] = args.mask_list

    # Run processing
    print("\nStarting processing...")
    results = process(**process_kwargs)

    print(f"\nProcessing complete!")
    print(f"Results saved to: {args.output_dir}")
    
