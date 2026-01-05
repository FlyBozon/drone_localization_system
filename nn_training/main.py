import numpy as np
import matplotlib.pyplot as plt
import cv2
import PIL
import os
import random
#later check if all libs needed, as were copyied from old files
import tensorflow as tf
from tensorflow import keras
# import segmentation_models as sm
# from tensorflow.keras.metrics import MeanIoU
# from sklearn.preprocessing import MinMaxScaler
# from keras.utils import to_categorical
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

from processor import *
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train semantic segmentation model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (uavid, landcover.ai, inria, deepglobe)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to datasets_info.json configuration file')

    # Model architecture
    parser.add_argument('--architecture', type=str, default='fpn',
                        choices=['unet', 'fpn', 'linknet', 'pspnet', 'deeplabv3', 'deeplabv3plus'],
                        help='Model architecture')
    parser.add_argument('--backbone', type=str, default='efficientnetb3',
                        help='Backbone architecture (e.g., efficientnetb3, resnet50, mobilenetv2)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--patch-size', type=int, default=256,
                        help='Size of patches for training')

    # ClearML
    parser.add_argument('--project-name', type=str, default='Segmentation',
                        help='ClearML project name')
    parser.add_argument('--task-name', type=str, default=None,
                        help='ClearML task name (default: {dataset}_training)')
    parser.add_argument('--no-clearml', action='store_true',
                        help='Disable ClearML integration')

    # Data preprocessing options
    parser.add_argument('--preprocess-tiles', action='store_true',
                        help='Create tiles from images (run into_tiles)')
    parser.add_argument('--tile-overlap', type=int, default=0,
                        help='Overlap size for tiles')
    parser.add_argument('--choose-useful', action='store_true',
                        help='Filter useful patches (run choose_useful)')
    parser.add_argument('--usefulness-threshold', type=float, default=0.05,
                        help='Usefulness threshold for choosing useful patches')
    parser.add_argument('--split-data', action='store_true',
                        help='Split data into train/val/test sets')

    # Pretrained model
    parser.add_argument('--pretrained-model', type=str, default=None,
                        help='Path to pretrained model for transfer learning')

    args = parser.parse_args()

    # Initialize ClearML if enabled
    if not args.no_clearml:
        from clearml import Task
        task_name = args.task_name if args.task_name else f"{args.dataset}_training"
        task = Task.init(project_name=args.project_name, task_name=task_name)

    # Initialize processor
    print(f"Initializing DatasetProcessor for dataset: {args.dataset}")
    processor = DatasetProcessor(args.dataset, dataset_info_path=args.config)
    processor.patch = args.patch_size

    # Preprocessing steps
    if args.preprocess_tiles:
        print(f"Creating {args.patch_size}x{args.patch_size} tiles...")
        processor.into_tiles(args.patch_size, overlap_size=args.tile_overlap)

    if args.choose_useful:
        print(f"Filtering useful patches (threshold: {args.usefulness_threshold})...")
        processor.choose_useful(usefulness_percent=args.usefulness_threshold)

    if args.split_data:
        print("Splitting data into train/val/test sets...")
        processor.divide_train_val_test()

    # Dataset-specific preprocessing
    if args.dataset.lower() == "uavid":
        print("Running UAVid-specific preprocessing...")
        processor.uavid_data_preprocess()

    # Load pretrained model if specified
    if args.pretrained_model:
        print(f"Loading pretrained model from: {args.pretrained_model}")
        import tensorflow as tf
        model = tf.keras.models.load_model(args.pretrained_model, compile=False)
        # Optionally freeze layers (uncomment if needed)
        # for layer in model.layers[:-2]:
        #     layer.trainable = False
        processor.model = model
        processor.BACKBONE = args.backbone
        print("Pretrained model loaded successfully")

    # Setup model
    print(f"Setting up {args.architecture.upper()} model with {args.backbone} backbone...")
    if args.pretrained_model is None:
        processor.setup_model(args.architecture, args.backbone)

    # Set training parameters
    print(f"Setting training parameters: epochs={args.epochs}, batch_size={args.batch_size}")
    processor.set_training_parameters(epochs=args.epochs, batch_size=args.batch_size)

    # Train model
    print("Starting training...")
    processor.train()

    # Plot statistics
    print("Plotting training statistics...")
    processor.plot_statistics()

    print("Training completed!")