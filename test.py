#fast code prototyping (needed to test how nn works for futher actions), written using claude ai

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
        
        # Calculate metrics
        # if true_mask is not None:
        #     iou = self.calculate_iou(true_mask, prediction_mask)
        #     pixel_acc = self.calculate_pixel_accuracy(true_mask, prediction_mask)
        #     results['iou'] = iou
        #     results['pixel_accuracy'] = pixel_acc
        if true_mask is not None:
            full_metrics = self.calculate_all_metrics(true_mask, prediction_mask)
            results["metrics"] = full_metrics
            print("hello world")
        
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
        """Visualize prediction results - saves raw mask, colored mask, and comparison"""
        img_name = os.path.basename(save_path).split('.')[0] if save_path else "image"
        
        if save_path:
            # Create organized folder structure
            base_dir = os.path.dirname(os.path.dirname(save_path))  # Go up to get base output dir
            
            raw_dir = os.path.join(base_dir, "raw")
            colored_dir = os.path.join(base_dir, "colored")
            comparison_dir = os.path.join(base_dir, "comparison")
            
            os.makedirs(raw_dir, exist_ok=True)
            os.makedirs(colored_dir, exist_ok=True)
            os.makedirs(comparison_dir, exist_ok=True)
            
            # Save raw prediction mask (grayscale)
            raw_path = os.path.join(raw_dir, f"{img_name}_raw.png")
            cv2.imwrite(raw_path, results['prediction'].astype(np.uint8))
            print(f"Saved raw mask to: {raw_path}")
            
            # Save colored prediction mask
            pred_colored = self.create_colored_mask(results['prediction'])
            colored_path = os.path.join(colored_dir, f"{img_name}_colored.png")
            cv2.imwrite(colored_path, cv2.cvtColor(pred_colored, cv2.COLOR_RGB2BGR))
            print(f"Saved colored mask to: {colored_path}")
            
            # Create and save comparison visualization
            n_cols = 3 if results['ground_truth'] is not None else 2
            fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
            
            if n_cols == 2:
                axes = [axes[0], axes[1]]
            else:
                axes = list(axes)
            
            # Original image
            axes[0].imshow(results['original'])
            axes[0].set_title(f'Oryginalny obraz {img_name}')
            axes[0].axis('off')

            # Ground truth (if available)
            if results['ground_truth'] is not None:
                gt_colored = self.create_colored_mask(results['ground_truth'])
                axes[1].imshow(gt_colored)
                axes[1].set_title('Rzeczywiste oznaczenia (GT)')
                axes[1].axis('off')
                
                # Prediction
                axes[2].imshow(pred_colored)
                title = 'Predykcja'
                if 'metrics' in results and 'mean_iou' in results['metrics']:
                    title += f"\nIoU: {results['metrics']['mean_iou']:.3f}"
                axes[2].set_title(title)
                axes[2].axis('off')
            else:
                # No GT - just show prediction
                axes[1].imshow(pred_colored)
                title = 'Predykcja'
                axes[1].set_title(title)
                axes[1].axis('off')
            
            plt.tight_layout()
            
            comparison_path = os.path.join(comparison_dir, f"{img_name}_comparison.png")
            plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
            print(f"Saved comparison visualization to: {comparison_path}")
            
            plt.close()  # Close figure to free memory


    def calculate_patch_significance(self, patch, pred_mask, gt_mask=None):
        """
        Calculate patch significance:
        - Base significance = percentage of non-background pixels.
        - Bonus factor for class diversity = more non-background classes -> higher score.
        """
        # Base: percentage of non-background pixels (same logic as original)
        non_background = np.sum(pred_mask != 0)
        total = pred_mask.size
        base_significance = (non_background / total) * 100

        # Count unique non-background classes
        unique_classes = np.unique(pred_mask)
        non_bg_classes = unique_classes[unique_classes != 0]
        num_classes = len(non_bg_classes)

        # If no non-background classes, significance stays 0
        if num_classes == 0:
            return 0.0

        # Class diversity multiplier (linear example)
        # 1 class -> x1.0   |  2 classes -> x1.3   |  3 classes -> x1.6, etc.
        diversity_multiplier = 1.0 + 0.3 * (num_classes - 1)

        # Final score: original logic enhanced with diversity
        significance = base_significance * diversity_multiplier
        return significance


    # def calculate_patch_significance(self, patch, pred_mask, gt_mask=None):
    #     """
    #     Calculate how 'significant' a patch is based on class distribution.
    #     Returns percentage of non-background pixels (classes != 0).
    #     """
    #     # Assume class 0 is background/empty
    #     non_background = np.sum(pred_mask != 0)
    #     total = pred_mask.size
    #     significance = (non_background / total) * 100
    #     return significance


    def save_interesting_patches(self, patches_data, patch_save_dir, max_patches=50, min_significance=75):
        print(f"\nFiltering patches by significance (>{min_significance}% non-background)...")
        
        # Calculate significance for each patch
        scored_patches = []
        for data in patches_data:
            significance = self.calculate_patch_significance(
                data['patch'], 
                data['pred_mask'],
                data.get('gt_mask')
            )
            data['significance'] = significance
            if significance >= min_significance:
                scored_patches.append(data)
        
        # Sort by significance (highest first)
        scored_patches.sort(key=lambda x: x['significance'], reverse=True)
        
        # Take top N patches
        selected_patches = scored_patches[:max_patches]
        
        print(f"Found {len(scored_patches)} patches with >{min_significance}% significance")
        print(f"Saving top {len(selected_patches)} patches...")
        
        # Save each patch as a 3-panel visualization
        for i, data in enumerate(selected_patches):
            patch = data['patch']
            pred_mask = data['pred_mask']
            gt_mask = data.get('gt_mask')
            position = data['position']
            significance = data['significance']
            
            # Create 3-panel figure
            n_cols = 3 if gt_mask is not None else 2
            fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 4))
            
            if n_cols == 2:
                axes = [axes[0], axes[1]]
            else:
                axes = list(axes)
            
            # Original patch
            axes[0].imshow(patch)
            axes[0].set_title('Oryginalny patch')
            axes[0].axis('off')
            
            # Ground truth (if available)
            if gt_mask is not None:
                gt_colored = self.create_colored_mask(gt_mask)
                axes[1].imshow(gt_colored)
                axes[1].set_title('Rzeczywiste oznaczenia (GT)')
                axes[1].axis('off')
                
                # Prediction
                pred_colored = self.create_colored_mask(pred_mask)
                axes[2].imshow(pred_colored)
                title = 'Predykcja'
                mean_iou = data.get('metrics', {}).get('mean_iou', None)
                if mean_iou is not None:
                    title += f"\nIoU: {mean_iou:.3f}"
                axes[2].set_title(title)
                axes[2].axis('off')
            else:
                pred_colored = self.create_colored_mask(pred_mask)
                axes[1].imshow(pred_colored)
                title = 'Predykcja'
                mean_iou = data.get('metrics', {}).get('mean_iou', None)
                if mean_iou is not None:
                    title += f"\nIoU: {mean_iou:.3f}"
                else:
                    title += "\nIoU: N/A"
                axes[1].set_title(title)
                axes[1].axis('off')
            
            # Add significance to title
            # fig.suptitle(f'Patch {i+1}/{len(selected_patches)} | Position: {position} | Significance: {significance:.1f}%', 
            #             fontsize=10)
            
            plt.tight_layout()
            
            # Save patch visualization
            start_h, start_w, _, _ = position
            patch_filename = os.path.join(patch_save_dir, f"patch_{i+1:03d}_row{start_h}_col{start_w}_sig{significance:.0f}.png")
            plt.savefig(patch_filename, dpi=100, bbox_inches='tight')
            plt.close()
            
            if (i + 1) % 10 == 0:
                print(f"Saved {i+1}/{len(selected_patches)} patches...")
        
        print(f"Finished saving {len(selected_patches)} interesting patches to {patch_save_dir}")


    def test_specific_image(self, img_path, mask_path=None, save_path=None, overlap=64, 
                        patch_save_dir=None, batch_size=16, save_patches=True, max_patches=50):
        """Predict on a single image with patchification"""
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
        if patch_save_dir and save_patches:
            os.makedirs(patch_save_dir, exist_ok=True)

        # Load ground truth if available
        true_mask = None
        if mask_path and os.path.exists(mask_path):
            true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            print(f"Loaded ground truth mask: {true_mask.shape}")

        patches_data = []  # Store patch data for later filtering

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
                if true_mask is not None:
                    true_mask = cv2.resize(
                        true_mask, 
                        (original_shape[1], original_shape[0]), 
                        interpolation=cv2.INTER_NEAREST
                    )
        
        else:
            # Extract patches
            patches, positions = self.extract_patches(img_rgb, overlap=overlap)
            
            # Extract GT patches if available
            gt_patches = None
            if true_mask is not None:
                if true_mask.shape != img_rgb.shape[:2]:
                    true_mask = cv2.resize(
                        true_mask, 
                        (img_rgb.shape[1], img_rgb.shape[0]), 
                        interpolation=cv2.INTER_NEAREST
                    )
                gt_patches, _ = self.extract_patches(true_mask, overlap=overlap)
            
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
            
            # Store patch data for filtering
            if save_patches and patch_save_dir:
                print(f"Collecting patch data for significance filtering...")
                for i, (patch, pred_probs, position) in enumerate(zip(patches, patch_predictions, positions)):
                    pred_mask = np.argmax(pred_probs, axis=-1)
                    
                    patch_data = {
                        'patch': patch,
                        'pred_mask': pred_mask,
                        'position': position,
                        'gt_mask': gt_patches[i] if gt_patches is not None else None
                    }
                    # Add metrics if GT is available
                    if gt_patches is not None and gt_patches[i] is not None:
                        patch_data['metrics'] = self.calculate_all_metrics(gt_patches[i], pred_mask)
                    patches_data.append(patch_data)
            
            # Combine all patches into full image prediction
            prediction_mask, prediction_probs = self.combine_patches(patch_predictions, positions, original_shape)
        
        # Resize true_mask to match prediction if needed
        if true_mask is not None and true_mask.shape != prediction_mask.shape:
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
        
        # Calculate metrics
        if true_mask is not None:
            full_metrics = self.calculate_all_metrics(true_mask, prediction_mask)
            results["metrics"] = full_metrics
        
        # Save interesting patches (top 50 with >50% significance)
        # if save_patches and patch_save_dir and len(patches_data) > 0:
        #     self.save_interesting_patches(
        #         patches_data, 
        #         patch_save_dir, 
        #         max_patches=max_patches,
        #         min_significance=50.0
        #     )
        
        return results

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
    
    # Get the dataset folder name from image path
    dataset_folder = os.path.basename(os.path.dirname(image_path))
    output_dir = os.path.join(output_dir, f"{dataset_folder}_output")
    os.makedirs(output_dir, exist_ok=True)
    
    image_name = os.path.basename(image_path).split('.')[0]
    
    # Create output directories - note the structure change
    save_path = os.path.join(output_dir, "temp", image_name)  # temp folder for internal use
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
    
    print(f"Results saved to: {output_dir}")
    return results

def _process_folder(tester, input_folder, output_dir, mask_folder=None, 
                    overlap=64, batch_size=16, extensions=None, save_patches=True, max_patches=50):
    if extensions is None:
        extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.tif', '.tiff']
    
    print(f"\nProcessing folder: {input_folder}")
    
    # Get the dataset folder name
    dataset_folder = os.path.basename(input_folder)
    output_dir = os.path.join(output_dir, f"{dataset_folder}_output")
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    # Prepare consolidated results structure
    from datetime import datetime
    consolidated_json_path = os.path.join(output_dir, "all_results.json")
    
    # Initialize or load existing results
    if os.path.exists(consolidated_json_path):
        with open(consolidated_json_path, 'r') as f:
            consolidated_results = json.load(f)
        print(f"Loaded existing results from {consolidated_json_path}")
    else:
        consolidated_results = {
            "model_name": os.path.splitext(os.path.basename(tester.model_path))[0],
            "model_path": tester.model_path,
            "dataset": tester.dataset_name,
            "n_classes": tester.n_classes,
            "patch_size": tester.patch_size,
            "overlap": overlap,
            "timestamp": datetime.now().isoformat(),
            "image_results": []
        }
    
    for i, img_path in enumerate(image_files):
        print(f"\n{'='*60}")
        print(f"Processing image {i+1}/{len(image_files)}: {os.path.basename(img_path)}")
        print(f"{'='*60}")
        
        image_name = os.path.basename(img_path).split('.')[0]
        original_ext = os.path.splitext(img_path)[1]
        
        # Find corresponding mask if mask_folder provided
        mask_path = None
        if mask_folder:
            for ext in extensions:
                potential_mask = os.path.join(mask_folder, f"{image_name}{ext}")
                if os.path.exists(potential_mask):
                    mask_path = potential_mask
                    break
        
        # Create output directories - modified structure
        save_path = os.path.join(output_dir, "temp", image_name)  # temp folder
        tile_patch_dir = os.path.join(output_dir, "patches", image_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_patches:
            os.makedirs(tile_patch_dir, exist_ok=True)
        
        # Process image
        try:
            results = tester.test_specific_image(
                img_path=img_path,
                mask_path=mask_path,
                save_path=save_path,
                overlap=overlap,
                patch_save_dir=tile_patch_dir if save_patches else None,
                batch_size=batch_size,
                save_patches=save_patches,
                max_patches=max_patches
            )
            
            if results is not None:
                # Visualize full image (saves to raw, colored, comparison folders)
                tester.visualize_prediction(results, save_path, tile_patch_dir)
                
                # Get paths for the organized outputs
                raw_path = os.path.join(output_dir, "raw", f"{image_name}_raw.png")
                colored_path = os.path.join(output_dir, "colored", f"{image_name}_colored.png")
                comparison_path = os.path.join(output_dir, "comparison", f"{image_name}_comparison.png")
                
                # Add to consolidated results
                image_result = {
                    "image_name": f"{image_name}{original_ext}",
                    "image_path": img_path,
                    "mask_path": mask_path,
                    "raw_mask_path": raw_path,
                    "colored_mask_path": colored_path,
                    "comparison_path": comparison_path,
                    "patches_dir": tile_patch_dir if save_patches else None,
                    "original_shape": list(results['original'].shape)
                }
                
                if "metrics" in results:
                    image_result["metrics"] = results["metrics"]
                
                consolidated_results["image_results"].append(image_result)
                
                # Save/update JSON after each image
                with open(consolidated_json_path, "w") as f:
                    json.dump(consolidated_results, f, indent=2)
                print(f"Updated consolidated results: {consolidated_json_path}")
                
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
    
    print(f"\nAll results saved to: {consolidated_json_path}")
    
    # Print summary
    _print_summary(results_list)
    
    return results_list

def _process_dataset(tester, dataset_root, output_dir, sequences='all', 
                     process_satellite=True, process_drone=True, overlap=64, batch_size=16):
    print(f"\nProcessing UAV_VisLoc_dataset from: {dataset_root}")
    
    # Get dataset name
    dataset_name = os.path.basename(dataset_root)
    output_dir = os.path.join(output_dir, f"{dataset_name}_output")
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    # Create batch output directory
    output_dir = os.path.join(output_dir, "batch_output")
    os.makedirs(output_dir, exist_ok=True)
    
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
    code_dir = "" #"/scratch/markryku/engineering_project/"
    data_dir = "" #"/data/markryku/"
    output_dir =  "output/" #"/data/markryku/output/" #"output/" 
    
    #MODEL_PATH = "trained_models/landcover.ai_fpn_efficientnetb3_100epochs_batch32_v1.keras" #"trained_models/landcover.ai_90_epochs_efficientnetb0_backbone_batch64_v1_early.keras" #"/data/markryku/output/final_models/landcover.ai_90_epochs_efficientnetb0_backbone_batch64_v1_early.keras"
    DATASET_CONFIG = f"{code_dir}nn_training/datasets_info.json"

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    MODEL_LIST = [
        #"trained_models/landcover.ai_fpn_efficientnetb3_100epochs_batch32_v1.keras",
        "trained_models/landcover.ai_90_epochs_efficientnetb0_backbone_batch64_v1_early.keras",
        #"trained_models/landcover.ai_101_epochs_resnet50_backbone_batch32_v1_early.keras",
        #"trained_models/landcover.ai_fpn_efficientnetb3_3epochs_batch32_v1.keras"
    ]
#     /data/markryku/output/final_models/v1/v1$ ls
# landcover.ai_90_epochs_efficientnetb0_backbone_batch64_v1_early.keras
# /data/markryku/output/final_models/v1/v1/landcover.ai_fpn_efficientnetb3_100epochs_batch32_v1.keras

    MODEL_PATH = "trained_models/landcover.ai_90_epochs_efficientnetb0_backbone_batch64_v1_early.keras" #"/data/markryku/output/final_models/v1/landcover.ai_90_epochs_efficientnetb0_backbone_batch64_v1_early.keras" #"trained_models/landcover.ai_unet_efficientnetb3_100epochs_batch32_v1.keras" #"trained_models/landcover.ai_90_epochs_efficientnetb0_backbone_batch64_v1_early.keras" #"/data/markryku/output/final_models/v1/v1/landcover.ai_fpn_efficientnetb3_100epochs_batch32_v1.keras" #"trained_models/landcover.ai_90_epochs_efficientnetb0_backbone_batch64_v1_early.keras"
     #"/data/markryku/output/final_models/v1/v1/landcover.ai_90_epochs_efficientnetb0_backbone_batch64_v1_early.keras" #"trained_models/landcover.ai_90_epochs_efficientnetb0_backbone_batch64_v1_early.keras"
    
    # process_multiple_models(
    #     model_paths=MODEL_LIST,
    #     dataset_config=DATASET_CONFIG,
    #     processing_mode="single",
    #     image_path="output_images/test.png",
    #     output_dir="comparison_output",
    #     overlap=64,
    #     batch_size=32
    # )

    # process_multiple_models(
    #     model_paths=MODEL_LIST,
    #     dataset_config=DATASET_CONFIG,
    #     processing_mode="single",
    #     image_path= "datasets/landcover.ai.v1/images/M-33-7-A-d-2-3.tif", #"output_images/lat52.229700_lon21.012200/geoportal_ortho.png",
    #     output_dir="comparison_output",
    #     overlap=64,
    #     batch_size=32
    # )

    # combine_model_outputs(
    #     output_root="comparison_output"
    #     #class_count=5  
    # )

    # ===== Process single image =====
    process(
        model_path=MODEL_PATH,
        dataset_config=DATASET_CONFIG,
        processing_mode="single",
        image_path= "output/patches/ortophoto/orthophoto/ref_map.png" #"output_images/lat52.229700_lon21.012200/geoportal_ortho.png", #"datasets/max/23.png", #"datasets/landcover.ai.v1/images/M-33-7-A-d-2-3.tif", #"output_images/lat52.229700_lon21.012200/geoportal_ortho.png", #f"{data_dir}datasets/uav-visloc/02/drone/02_0021.JPG",
        output_dir=f"{output_dir}single_test_v2",
        #mask_path = "datasets/landcover.ai.v1/masks/M-33-7-A-d-2-3.tif",
        overlap=64,
        batch_size=64  # Adjust based on your GPU memory
    )
    
    # ===== Process all images in a folder =====
    # process(
    #     model_path=MODEL_PATH,
    #     dataset_config=DATASET_CONFIG,
    #     processing_mode="folder",
    #     input_folder= "datasets/max", #f"{data_dir}datasets/landcover.ai.v1/images/", #"output_processed", #f"{data_dir}datasets/uav-visloc/02/drone",
    #     output_dir=f"{output_dir}folder_v2_test",
    #     #mask_folder = f"{data_dir}datasets/landcover.ai.v1/masks/",
    #     overlap=64,
    #     batch_size=32
    # )
    
    # ===== Process entire dataset =====
    # process(
    #     model_path=MODEL_PATH,
    #     dataset_config=DATASET_CONFIG,
    #     processing_mode="dataset",
    #     dataset_root= "datasets/UAV_VisLoc_dataset", #f"{data_dir}datasets/uav-visloc",
    #     output_dir=f"{output_dir}full_dataset1",
    #     sequences= 'all',  # or 'all' for all sequences
    #     process_satellite=True,
    #     process_drone=True,
    #     overlap=64,
    #     batch_size=32  # Start with 16-32, increase if you have more GPU memory
    # )
    
    # ===== Process specific list of images =====
    # image_list = [
    #     f"{data_dir}datasets/uav-visloc/01/drone/01_0001.JPG",
    #     f"{data_dir}datasets/uav-visloc/01/drone/01_0002.JPG",
    #     f"{data_dir}datasets/uav-visloc/02/satellite02.tif"
    # ]
    # process(
    #     model_path=MODEL_PATH,
    #     dataset_config=DATASET_CONFIG,
    #     processing_mode="batch",
    #     image_list=image_list,
    #     output_dir=f"{output_dir}batch_test",
    #     overlap=64,
    #     batch_size=32
    # )
