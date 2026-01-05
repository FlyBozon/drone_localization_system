import numpy as np
import tensorflow as tf
from processor import DatasetProcessor
import segmentation_models as sm
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Fine-tune semantic segmentation model on INRIA dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--pretrained-model', type=str, required=True,
                        help='Path to pretrained model to fine-tune')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to datasets_info.json configuration file')

    # Dataset
    parser.add_argument('--dataset', type=str, default='inria',
                        help='Dataset name')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=5e-5,
                        help='Learning rate for training')
    parser.add_argument('--freeze-ratio', type=float, default=0.5,
                        help='Ratio of encoder layers to freeze (0.0-1.0)')
    parser.add_argument('--building-idx', type=int, default=1,
                        help='Class index for building class')

    # Model
    parser.add_argument('--backbone', type=str, default='efficientnetb0',
                        help='Backbone architecture')

    # ClearML
    parser.add_argument('--project-name', type=str, default='Segmentation',
                        help='ClearML project name')
    parser.add_argument('--task-name', type=str, default='INRIA_fine_tuning',
                        help='ClearML task name')
    parser.add_argument('--no-clearml', action='store_true',
                        help='Disable ClearML integration')
    parser.add_argument('--phase', type=int, default=2,
                        help='Training phase number (for tracking)')

    args = parser.parse_args()

    # Initialize ClearML if enabled
    if not args.no_clearml:
        from clearml import Task
        task = Task.init(
            project_name=args.project_name,
            task_name=args.task_name
        )

        task.connect({
            'dataset': args.dataset.upper(),
            'phase': args.phase,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'freeze_ratio': args.freeze_ratio,
            'building_class_idx': args.building_idx,
            'backbone': args.backbone,
            'pretrained_model': args.pretrained_model
        })

    # Initialize processor
    print(f"Initializing DatasetProcessor for dataset: {args.dataset}")
    processor = DatasetProcessor(args.dataset, dataset_info_path=args.config)

    if not args.no_clearml:
        processor.task = task

    # Load pretrained model
    print(f"Loading pretrained model from: {args.pretrained_model}")
    model = tf.keras.models.load_model(args.pretrained_model, compile=False)
    processor.model = model
    processor.BACKBONE = args.backbone
    processor.preprocess_input = sm.get_preprocessing(processor.BACKBONE)
    print("Pretrained model loaded successfully")

    # Preprocess data
    print("Preprocessing INRIA dataset...")
    processor.preprocess_for_inria()

    # Set training parameters
    print(f"Setting training parameters: epochs={args.epochs}, batch_size={args.batch_size}")
    processor.set_training_parameters(epochs=args.epochs, batch_size=args.batch_size)

    # Train model
    print(f"Starting fine-tuning (lr={args.learning_rate}, freeze_ratio={args.freeze_ratio})...")
    history = processor.train_inria_multiclass(
        epochs=args.epochs,
        building_idx=args.building_idx,
        freeze_encoder=True,
        freeze_ratio=args.freeze_ratio,
        learning_rate=args.learning_rate
    )

    # Plot statistics
    print("Plotting training statistics...")
    processor.plot_statistics()

    print(f"\nTraining complete!")
    if not args.no_clearml:
        print(f"Check ClearML project '{args.project_name}' for results")