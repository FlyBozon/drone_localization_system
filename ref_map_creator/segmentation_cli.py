#!/usr/bin/env python3
"""
Segmentation script - CLI version
Accepts image path and output folder from command line
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import cv2
import os
import sys
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
import segmentation_models as sm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF warnings

print(f"TensorFlow version: {tf.__version__}")
print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("WARNING: No GPU detected! Running on CPU (this will be slower).")


# Fix keras.backend compatibility
import keras.backend as K
if not hasattr(K, 'sigmoid'):
    K.sigmoid = tf.nn.sigmoid
if not hasattr(K, 'swish'):
    K.swish = lambda x: x * tf.nn.sigmoid(x)
if not hasattr(K, 'relu'):
    K.relu = tf.nn.relu


class LandCoverTester:
    def __init__(self, model_path, dataset_config_path="datasets_info.json", dataset_name="landcover.ai"):
        self.model_path = model_path
        self.dataset_name = dataset_name
        
        # Load dataset config
        self._load_dataset_config(dataset_config_path)
        
        # CRITICAL: Match training preprocessing
        self.BACKBONE = 'efficientnetb0'
        self.preprocess_input = sm.get_preprocessing(self.BACKBONE)
        self.patch_size = 256
        
        # Load model
        self.model = None
        self._load_model()
        
    def _load_dataset_config(self, config_path):
        """Load dataset configuration"""
        if not os.path.exists(config_path):
            print(f"Warning: Config file not found: {config_path}")
            print("Using default configuration...")
            self.n_classes = 5
            self.class_names = ['Background', 'Buildings', 'Woodlands', 'Water', 'Roads']
            return
            
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
        """Load trained model"""
        print(f"Loading model from: {self.model_path}")
        
        custom_objects = {
            'iou_score': sm.metrics.iou_score,
            'f1-score': sm.metrics.f1_score,
        }
        
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
        """Preprocessing matching training"""
        img = img.astype(np.float32) / 255.0
        img = self.preprocess_input(img)
        return img
    
    def extract_patches(self, image, overlap=64):
        """Extract overlapping patches"""
        h, w = image.shape[:2]
        step = self.patch_size - overlap
        
        patches = []
        positions = []
        
        print(f"Image size: {h}x{w}, Patch size: {self.patch_size}x{self.patch_size}, Overlap: {overlap}")
        
        n_patches_h = (h - self.patch_size) // step + 1
        n_patches_w = (w - self.patch_size) // step + 1
        
        if (n_patches_h - 1) * step + self.patch_size < h:
            n_patches_h += 1
        if (n_patches_w - 1) * step + self.patch_size < w:
            n_patches_w += 1
        
        print(f"Extracting {n_patches_h}x{n_patches_w} = {n_patches_h * n_patches_w} patches...")
        
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                start_h = min(i * step, h - self.patch_size)
                start_w = min(j * step, w - self.patch_size)
                end_h = start_h + self.patch_size
                end_w = start_w + self.patch_size
                
                patch = image[start_h:end_h, start_w:end_w]
                
                if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
                    print(f"Warning: Patch {i},{j} has incorrect size: {patch.shape}")
                    continue
                
                patches.append(patch)
                positions.append((start_h, start_w, end_h, end_w))
        
        return patches, positions
    
    def combine_patches(self, predictions, positions, original_shape):
        """Combine overlapping patches"""
        h, w = original_shape[:2]
        
        combined_probs = np.zeros((h, w, self.n_classes), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)
        
        print(f"Combining {len(predictions)} patches into {h}x{w} image...")
        
        for pred_probs, (start_h, start_w, end_h, end_w) in zip(predictions, positions):
            combined_probs[start_h:end_h, start_w:end_w] += pred_probs
            weight_map[start_h:end_h, start_w:end_w] += 1.0
        
        weight_map[weight_map == 0] = 1.0
        combined_probs = combined_probs / weight_map[:, :, np.newaxis]
        prediction_mask = np.argmax(combined_probs, axis=-1)
        
        return prediction_mask, combined_probs
    
    def predict_single_image(self, img_path, overlap=64, batch_size=16):
        """Predict on single image - GPU OPTIMIZED"""
        print(f"\n{'='*60}")
        print(f"Processing: {img_path}")
        print(f"{'='*60}")
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            return None
        
        print(f"Loaded image: {img.shape}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_img = img_rgb.copy()
        original_shape = img_rgb.shape

        if img_rgb.shape[0] <= self.patch_size and img_rgb.shape[1] <= self.patch_size:
            print("Image smaller than patch size, processing as single patch...")
            
            if img_rgb.shape[0] != self.patch_size or img_rgb.shape[1] != self.patch_size:
                img_rgb_resized = cv2.resize(img_rgb, (self.patch_size, self.patch_size))
            else:
                img_rgb_resized = img_rgb
            
            img_preprocessed = self.preprocess_image(img_rgb_resized)
            img_batch = np.expand_dims(img_preprocessed, axis=0)
            prediction_probs = self.model.predict(img_batch, verbose=0)[0]
            prediction_mask = np.argmax(prediction_probs, axis=-1)

            if original_shape[:2] != (self.patch_size, self.patch_size):
                prediction_mask = cv2.resize(
                    prediction_mask.astype(np.uint8), 
                    (original_shape[1], original_shape[0]), 
                    interpolation=cv2.INTER_NEAREST
                )
        else:
            print("Large image - using patchification...")
            
            patches, positions = self.extract_patches(img_rgb, overlap=overlap)
            
            print(f"Preprocessing {len(patches)} patches...")
            preprocessed_patches = []
            for patch in patches:
                preprocessed_patches.append(self.preprocess_image(patch))
            
            print(f"Running inference on {len(preprocessed_patches)} patches (batch_size={batch_size})...")
            all_predictions = []
            
            for i in range(0, len(preprocessed_patches), batch_size):
                batch = preprocessed_patches[i:i+batch_size]
                batch_array = np.array(batch)
                
                predictions = self.model.predict(batch_array, verbose=0)
                all_predictions.extend(predictions)
                
                if (i // batch_size + 1) % 10 == 0 or i + batch_size >= len(preprocessed_patches):
                    progress = min(i + batch_size, len(preprocessed_patches))
                    print(f"  Progress: {progress}/{len(preprocessed_patches)} patches", end='\r', flush=True)
                    print("." * 3, end='', flush=True)
            
            print()  # New line after progress
            
            prediction_mask, _ = self.combine_patches(all_predictions, positions, original_shape)
        
        print(f"Prediction complete!")
        print(f"Unique classes in prediction: {np.unique(prediction_mask)}")
        
        return {
            'original_image': original_img,
            'prediction_mask': prediction_mask
        }


def process_image(model_path, image_path, output_folder, overlap=64, batch_size=32):
    """Process single image and save results"""
    
    # Find dataset config
    config_path = "datasets_info.json"
    possible_paths = [
        config_path,
        os.path.join("nn_training", config_path),
        os.path.join("..", config_path),
        os.path.join("..", "nn_training", config_path)
    ]
    
    config_found = None
    for path in possible_paths:
        if os.path.exists(path):
            config_found = path
            break
    
    if not config_found:
        print(f"Warning: datasets_info.json not found, using defaults")
        config_found = config_path
    
    # Initialize tester
    print("Initializing segmentation model...")
    tester = LandCoverTester(
        model_path=model_path,
        dataset_config_path=config_found
    )
    
    # Process
    results = tester.predict_single_image(
        img_path=image_path,
        overlap=overlap,
        batch_size=batch_size
    )
    
    if results is None:
        print("ERROR: Processing failed")
        return False
    
    # Save results
    print(f"\nSaving results to: {output_folder}/")
    os.makedirs(output_folder, exist_ok=True)
    
    # Save raw prediction mask
    mask_path = os.path.join(output_folder, "segmentation_nn_raw.png")
    cv2.imwrite(mask_path, results['prediction_mask'])
    print(f"  Saved raw mask: {mask_path}")
    
    # Save colored visualization
    colors = {
        0: (0, 0, 0),         # Background - black
        1: (255, 0, 0),       # Buildings - red
        2: (0, 255, 0),       # Woodlands - green
        3: (0, 0, 255),       # Water - blue
        4: (128, 128, 128)    # Roads - gray
    }
    
    colored_mask = np.zeros((*results['prediction_mask'].shape, 3), dtype=np.uint8)
    for class_id, color in colors.items():
        colored_mask[results['prediction_mask'] == class_id] = color
    
    colored_path = os.path.join(output_folder, "segmentation_nn_colored.png")
    cv2.imwrite(colored_path, cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
    print(f"  Saved colored mask: {colored_path}")
    
    # Save visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(results['original_image'])
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(results['prediction_mask'], cmap='gray', vmin=0, vmax=4)
    axes[1].set_title('Segmentation Mask (Raw)')
    axes[1].axis('off')
    
    axes[2].imshow(colored_mask)
    axes[2].set_title('Segmentation Mask (Colored)')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    viz_path = os.path.join(output_folder, "segmentation_nn_visualization.png")
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"  Saved visualization: {viz_path}")
    
    plt.close(fig)
    
    # Print statistics
    print(f"\nSegmentation Statistics:")
    unique, counts = np.unique(results['prediction_mask'], return_counts=True)
    total = results['prediction_mask'].size
    
    class_names = ['Background', 'Buildings', 'Woodlands', 'Water', 'Roads']
    for class_id, count in zip(unique, counts):
        name = class_names[class_id] if class_id < len(class_names) else f'Class_{class_id}'
        percentage = (count / total) * 100
        print(f"  {name}: {count} pixels ({percentage:.2f}%)")
    
    print("SEGMENTATION COMPLETE!")
    
    return True


if __name__ == "__main__":
    # Default model path
    MODEL_PATH = "/home/flybozon/engineering_project/trained_models/landcover.ai_90_epochs_efficientnetb0_backbone_batch64_v1_early.keras"
    
    # Alternative model paths to try
    alternative_paths = [
        MODEL_PATH,
        os.path.join("nn_training", MODEL_PATH),
        os.path.join("..", MODEL_PATH),
        os.path.join("..", "nn_training", MODEL_PATH),
        "model.keras",
        "trained_model.keras"
    ]
    
    model_found = None
    for path in alternative_paths:
        if os.path.exists(path):
            model_found = path
            break
    
    if not model_found:
        print("ERROR: Model file not found!")
        print(f"Searched paths: {alternative_paths}")
        print("\nPlease ensure the model file exists at one of these locations:")
        print("  - trained_models/landcover.ai_90_epochs_efficientnetb0_backbone_batch64_v1_early.keras")
        print("  - model.keras")
        sys.exit(1)
    
    print(f"Using model: {model_found}")
    
    if len(sys.argv) >= 3:
        # CLI mode
        image_path = sys.argv[1]
        output_folder = sys.argv[2]
        overlap = int(sys.argv[3]) if len(sys.argv) > 3 else 64
        batch_size = int(sys.argv[4]) if len(sys.argv) > 4 else 32
        
        if not os.path.exists(image_path):
            print(f"ERROR: Image not found: {image_path}")
            sys.exit(1)
        
        success = process_image(
            model_path=model_found,
            image_path=image_path,
            output_folder=output_folder,
            overlap=overlap,
            batch_size=batch_size
        )
        
        sys.exit(0 if success else 1)
    else:
        print("Usage: python segmentation_cli.py <image_path> <output_folder> [overlap] [batch_size]")
        print("\nExample:")
        print("  python segmentation_cli.py lat_54_371503_lon_18_618262/image_orto.jpg lat_54_371503_lon_18_618262 64 32")
        sys.exit(1)
