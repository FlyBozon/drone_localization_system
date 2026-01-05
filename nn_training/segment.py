import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os
import json
from tensorflow.keras.models import load_model
import tensorflow as tf
import segmentation_models as sm
from sklearn.preprocessing import MinMaxScaler
import tensorflow.keras.backend as K

class SegmentationInference:
    def __init__(self, model_path, images_dir, output_dir, dataset_config_path="datasets_info.json"):
        self.model_path = model_path
        self.images_dir = images_dir
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
                
            print(f"Loaded config for {self.dataset_name}: {self.n_classes} classes")
            print(f"Patch size: {self.patch_size}, Overlap: {self.overlap}")
            
        except FileNotFoundError:
            print(f"Warning: Config file {self.dataset_config_path} not found. Using default parameters.")
            self.n_classes = 8  # Default for UAVid
            
    def _load_model(self):
        """Load the trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
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
                # Loading full .keras model (recommended format)
                print("Loading .keras model...")
                with tf.keras.utils.custom_object_scope(custom_objects):
                    self.model = load_model(self.model_path, custom_objects=custom_objects, compile=False)
                print("Model loaded successfully!")
                
            elif self.model_path.endswith('.h5') and 'weights' in self.model_path:
                # Loading weights file
                print("Loading model weights...")
                # Create model architecture first
                model_args = {
                    'backbone_name': self.backbone,
                    'encoder_weights': None,  # Don't load ImageNet weights
                    'input_shape': (self.patch_size, self.patch_size, 3),
                    'classes': self.n_classes,
                    'activation': 'softmax'
                }
                
                # Assuming U-Net architecture (adjust as needed)
                self.model = sm.Unet(**model_args)
                self.model.load_weights(self.model_path)
                print("Model weights loaded successfully!")
                
            else:
                # Try loading as full model (for other formats)
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
            # First, try to load just to get the architecture info
            temp_model = load_model(self.model_path, compile=False)
            
            # Extract key information
            input_shape = temp_model.input_shape[1:]  # Remove batch dimension
            output_shape = temp_model.output_shape[-1]  # Number of classes
            
            del temp_model  # Clean up
            
            print(f"Detected input shape: {input_shape}, output classes: {output_shape}")
            
            # Recreate model with segmentation_models
            model_args = {
                'backbone_name': self.backbone,
                'encoder_weights': 'imagenet',
                'input_shape': input_shape,
                'classes': output_shape,
                'activation': 'softmax'
            }
            
            # Try different architectures
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
                    
                    # Try to get weights from original model
                    original_model = load_model(self.model_path, compile=False)
                    
                    # Copy weights layer by layer where possible
                    for i, layer in enumerate(new_model.layers):
                        if i < len(original_model.layers):
                            try:
                                layer.set_weights(original_model.layers[i].get_weights())
                            except:
                                pass  # Skip layers where weights don't match
                    
                    self.model = new_model
                    print(f"Successfully loaded with {arch} architecture!")
                    return
                    
                except Exception as e:
                    print(f"{arch} failed: {e}")
                    continue
            
            print("All architectures failed, using original model anyway...")
            # Load original model and hope for the best
            self.model = load_model(self.model_path, compile=False)
            
        except Exception as e:
            print(f"Alternative loading failed: {e}")
            raise
    
    def _setup_color_mapping(self):
        """Setup color mapping for visualization"""
        if self.dataset_name == 'uavid' and hasattr(self, 'class_colors_dict'):
            # Use UAVid color mapping
            self.class_colors = {}
            for i, (class_name, class_id) in enumerate(zip(self.class_names, self.class_ids)):
                self.class_colors[class_id] = self.class_colors_dict[class_name]
             # Ensure RGB only
            for k, v in self.class_colors.items():
                self.class_colors[k] = np.array(v)[:3].tolist()

        else:
            # Default color mapping
            self.class_colors = {
                i: [int(c) for c in plt.cm.tab10(i)[:3] * 255] 
                for i in range(self.n_classes)
            }
        
        print(f"Color mapping setup for {len(self.class_colors)} classes")
    
    def preprocess_patch(self, patch):
        """Preprocess a single patch for model input"""
        # Normalize to 0-1 range
        patch = patch.astype(np.float32) / 255.0
        
        # Apply backbone-specific preprocessing
        patch = self.preprocess_input(patch)
        
        # Add batch dimension
        patch = np.expand_dims(patch, axis=0)
        
        return patch
    
    def extract_patches(self, image):
        """Extract overlapping patches from image"""
        h, w = image.shape[:2]
        step = self.patch_size - self.overlap
        
        patches = []
        positions = []
        
        # Calculate number of patches
        n_patches_h = (h - self.patch_size) // step + 1
        n_patches_w = (w - self.patch_size) // step + 1
        
        # Handle edge cases - ensure we cover the entire image
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
    
    # def combine_patches(self, predictions, positions, original_shape):
    #     """Combine patch predictions back into full image"""
    #     h, w = original_shape[:2]
        
    #     # Initialize output arrays
    #     combined_prediction = np.zeros((h, w, self.n_classes), dtype=np.float32)
    #     weight_map = np.zeros((h, w), dtype=np.float32)
        
    #     for pred, (start_h, start_w, end_h, end_w) in zip(predictions, positions):
    #         # Add prediction to combined result
    #         combined_prediction[start_h:end_h, start_w:end_w] += pred
    #         weight_map[start_h:end_h, start_w:end_w] += 1.0
        
    #     # Normalize by overlap count
    #     weight_map[weight_map == 0] = 1  # Avoid division by zero
    #     combined_prediction = combined_prediction / weight_map[:, :, np.newaxis]
        
    #     # Get final class predictions
    #     final_prediction = np.argmax(combined_prediction, axis=-1)
        
    #     return final_prediction, combined_prediction


    def combine_patches(self, predictions, positions, original_shape):
        """Combine patch predictions back into full image"""
        h, w = original_shape[:2]

        # INIT
        combined_prediction = np.zeros((h, w, self.n_classes), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)

        for pred, (start_h, start_w, end_h, end_w) in zip(predictions, positions):
            
            # ---- FIX: Ensure prediction has shape (H, W, C) ----
            if pred.ndim == 2:
                # Model returned class indices → one-hot
                pred_one_hot = np.zeros((*pred.shape, self.n_classes), dtype=np.float32)
                for c in range(self.n_classes):
                    pred_one_hot[:, :, c] = (pred == c).astype(np.float32)
                pred = pred_one_hot

            elif pred.ndim == 1:
                # Dead case: flat vector
                print("ERROR: Flat prediction detected, reshaping not possible")
                continue
            
            elif pred.ndim == 3 and pred.shape[-1] != self.n_classes:
                # Wrong channel count → resize channels
                print(f"WARNING: wrong channel count, resizing {pred.shape} → {self.n_classes}")
                pred = pred[:, :, :self.n_classes]

            patch_h = end_h - start_h
            patch_w = end_w - start_w

            # ---- FIX: resize pred to patch dimensions (just in case) ----
            if pred.shape[0] != patch_h or pred.shape[1] != patch_w:
                pred = cv2.resize(pred, (patch_w, patch_h), interpolation=cv2.INTER_LINEAR)

            # ACCUMULATE PATCH
            combined_prediction[start_h:end_h, start_w:end_w] += pred
            weight_map[start_h:end_h, start_w:end_w] += 1.0

        # NORMALIZE
        weight_map[weight_map == 0] = 1
        combined_prediction /= weight_map[:, :, None]

        # FINAL CLASS MAP
        final_prediction = np.argmax(combined_prediction, axis=-1)

        return final_prediction, combined_prediction

    
    def create_colored_mask(self, prediction):
        """Convert class prediction to colored mask"""
        h, w = prediction.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id, color in self.class_colors.items():
            mask = prediction == class_id
            colored_mask[mask] = color
        
        return colored_mask
    
    def create_transparent_overlay(self, original_image, colored_mask, alpha=0.7):
        """Create transparent overlay of segmentation on original image"""
        # Ensure both images have the same dtype
        original_image = original_image.astype(np.uint8)
        colored_mask = colored_mask.astype(np.uint8)
        
        # Create a mask where we have actual predictions (not background/class 0)
        # This assumes class 0 is background - adjust if needed
        has_prediction = np.any(colored_mask > 0, axis=2)
        
        # Create overlay only where we have predictions
        overlay = original_image.copy()
        
        # Apply blending only to areas with predictions
        for i in range(3):  # RGB channels
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
            if class_id == 0:  # Skip background class
                continue
                
            # Create mask for this specific class
            class_mask = (prediction == class_id)
            
            if np.any(class_mask):  # Only create if class exists in image
                # Create colored overlay for this class
                colored_overlay = np.zeros_like(original_image)
                #colored_overlay[class_mask] = color
                colored_overlay[class_mask] = color[:3]

                
                # Blend with original image
                overlay = original_image.copy()
                for i in range(3):  # RGB channels
                    overlay[:, :, i] = np.where(
                        class_mask,
                        (1 - alpha) * original_image[:, :, i] + alpha * colored_overlay[:, :, i],
                        original_image[:, :, i]
                    )
                
                class_name = self.class_names[class_id] if hasattr(self, 'class_names') else f"Class_{class_id}"
                overlays[class_name] = overlay.astype(np.uint8)
        
        return overlays
    
    def predict_image(self, image_path):
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
        # predictions = []
        # for i, patch in enumerate(patches):
        #     if i % 10 == 0:
        #         print(f"Processing patch {i+1}/{len(patches)}")
            
        #     # Preprocess patch
        #     preprocessed_patch = self.preprocess_patch(patch)
            
        #     # Predict
        #     pred = self.model.predict(preprocessed_patch, verbose=0)
        #     predictions.append(pred[0])  # Remove batch dimension

        
        # Predict on patches
        predictions = []
        for i, patch in enumerate(patches):
            if i % 10 == 0:
                print(f"Processing patch {i+1}/{len(patches)}")
            
            # Preprocess patch
            preprocessed_patch = self.preprocess_patch(patch)
            
            # Predict
            pred = self.model.predict(preprocessed_patch, verbose=0)[0]

            # ---- FIX: Ensure pred has correct shape ----
            if pred.ndim == 2:        # e.g. (H, W)
                pred = np.expand_dims(pred, axis=-1)     # → (H, W, 1)

            elif pred.ndim == 1:      # flat vector → impossible to place in image
                print("ERROR: flat prediction shape, skipping patch")
                continue

            # Add corrected prediction
            predictions.append(pred)

        # Combine patches
        final_prediction, confidence_map = self.combine_patches(predictions, positions, original_shape)
        
        # Create visualizations
        colored_mask = self.create_colored_mask(final_prediction)
        overlay = self.create_transparent_overlay(image, colored_mask, alpha=0.7)
        
        # Create class-specific overlays for detailed analysis
        class_overlays = self.create_class_specific_overlays(image, final_prediction, alpha=0.7)
        
        return {
            'prediction': final_prediction,
            'colored_mask': colored_mask,
            'overlay': overlay,
            'class_overlays': class_overlays,
            'confidence_map': confidence_map,
            'original_image': image
        }
    
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
        
        # Create a combined visualization showing original + overlay side by side
        combined_viz = self.create_side_by_side_comparison(results['original_image'], results['overlay'])
        combined_path = f"{self.output_dir}/overlays/{base_name}_comparison.png"
        cv2.imwrite(combined_path, combined_viz)
        
        print(f"Results saved for {base_name}")
        print(f"  - Main overlay: {overlay_path}")
        print(f"  - Class-specific overlays: {len(class_overlay_paths)} classes")
        print(f"  - Side-by-side comparison: {combined_path}")
        
        return {
            'prediction_path': pred_path,
            'colored_path': colored_path,
            'overlay_path': overlay_path,
            'class_overlay_paths': class_overlay_paths,
            'comparison_path': combined_path
        }
    
    def create_side_by_side_comparison(self, original, overlay):
        """Create side-by-side comparison of original and overlay"""
        # Ensure both images are the same height
        h, w = original.shape[:2]
        
        # Create combined image
        combined = np.zeros((h, w*2, 3), dtype=np.uint8)
        combined[:, :w] = original
        combined[:, w:] = overlay
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = min(h, w) / 1000  # Scale font based on image size
        thickness = max(1, int(font_scale * 2))
        
        cv2.putText(combined, "Original", (10, 30), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(combined, "Segmentation Overlay", (w + 10, 30), font, font_scale, (255, 255, 255), thickness)
        
        return combined
    
    def process_directory(self):
        """Process all images in the input directory"""
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(self.images_dir, ext)))
            image_files.extend(glob.glob(os.path.join(self.images_dir, ext.upper())))
        
        if not image_files:
            print(f"No images found in {self.images_dir}")
            return
        
        print(f"Found {len(image_files)} images to process")
        
        # Process each image
        results_summary = []
        
        for i, image_path in enumerate(image_files):
            print(f"\n--- Processing image {i+1}/{len(image_files)} ---")
            
            try:
                # Run prediction
                results = self.predict_image(image_path)
                
                if results is not None:
                    # Save results
                    image_name = os.path.basename(image_path)
                    saved_paths = self.save_results(results, image_name)
                    
                    results_summary.append({
                        'image_name': image_name,
                        'original_path': image_path,
                        'saved_paths': saved_paths,
                        'success': True
                    })
                else:
                    results_summary.append({
                        'image_name': os.path.basename(image_path),
                        'original_path': image_path,
                        'success': False,
                        'error': 'Failed to process image'
                    })
                    
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results_summary.append({
                    'image_name': os.path.basename(image_path),
                    'original_path': image_path,
                    'success': False,
                    'error': str(e)
                })
        
        # Print summary
        successful = sum(1 for r in results_summary if r['success'])
        print(f"\nProcessing Summary:")
        print(f"Total images: {len(image_files)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(image_files) - successful}")
        print(f"Results saved to: {self.output_dir}")
        
        return results_summary
    
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


# Main execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Run semantic segmentation inference on images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model (.keras file)')
    parser.add_argument('--images-dir', type=str, required=True,
                        help='Directory containing input images')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save results')

    # Optional arguments
    parser.add_argument('--config', type=str, default='datasets_info.json',
                        help='Path to dataset configuration JSON file')
    parser.add_argument('--dataset-name', type=str, default='uavid',
                        help='Dataset name (uavid, landcover.ai, etc.)')
    parser.add_argument('--backbone', type=str, default='efficientnetb3',
                        help='Model backbone architecture')
    parser.add_argument('--patch-size', type=int, default=512,
                        help='Size of patches for processing')
    parser.add_argument('--overlap', type=int, default=64,
                        help='Overlap between patches')
    parser.add_argument('--num-samples', type=int, default=3,
                        help='Number of sample results to visualize')

    args = parser.parse_args()

    # Initialize inference
    print("Initializing segmentation inference...")
    print(f"Model: {args.model_path}")
    print(f"Images: {args.images_dir}")
    print(f"Output: {args.output_dir}")

    inference = SegmentationInference(
        model_path=args.model_path,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        dataset_config_path=args.config
    )

    # Override configuration if specified
    if args.patch_size != 512:
        inference.patch_size = args.patch_size
    if args.overlap != 64:
        inference.overlap = args.overlap
    if args.dataset_name != 'uavid':
        inference.dataset_name = args.dataset_name
    if args.backbone != 'efficientnetb3':
        inference.backbone = args.backbone
        inference.preprocess_input = sm.get_preprocessing(args.backbone)

    # Process all images in directory
    print("Starting batch processing...")
    results = inference.process_directory()

    # Visualize some results
    print("Visualizing sample results...")
    inference.visualize_sample_results(num_samples=args.num_samples)

    print("Inference completed!")