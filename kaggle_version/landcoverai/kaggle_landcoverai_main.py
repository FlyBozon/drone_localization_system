import numpy as np
import matplotlib.pyplot as plt
import cv2
import PIL
# import os
import random
import tensorflow as tf
from tensorflow import keras
from clearml import Task

from processor import *


# import segm models like that on kaggle, other way wouldnt work
import os
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import segmentation_models as sm


if Task.current_task():
    Task.current_task().close()

task = Task.init(
    project_name="Inzynierka-landcover.ai",
    task_name="Run 3",
    task_type=Task.TaskTypes.training
)

INPUT_dataset_DIR = '/kaggle/input/landcover-ai-v1/'  
INPUT_DIR = '/kaggle/input/' 
OUTPUT_DIR = '/kaggle/working/'  


datasets_info = f'{INPUT_DIR}infodatasets/datasets_info.json'
print(datasets_info)
current_dataset = "landcover.ai"

processor = DatasetProcessor(current_dataset, datasets_info)

processor.into_tiles(256, 0)
processor.choose_useful(0.05)
processor.divide_train_val_test()

processor.setup_model('resnet34')

processor.train()  

processor.plot_statistics()