import sys
import os
sys.path.append("../..")


from src.utils import configurations
from ml_collections import config_dict
import tensorflow as tf


def load_dataset(dataset: configurations.DataSets, configs: config_dict) -> list:
    #prepare the train and test file paths
    train_dir = os.path.join(dataset, 'train')
    test_dir = os.path.join(dataset, 'test')
    
    #prepare the class names
    class_names = ['angry', 'disgust', 'fear', 'happy', 'sad','surprise','neutral']
    
    
    #split the train set into validation and train sets and load them into a tf.data.Dataset
    train_set, validation_set = tf.keras.utils.image_dataset_from_directory(
        directory=train_dir,
        labels='inferred',
        label_mode='int',
        class_names=class_names,
        color_mode='rgb',
        batch_size=32,
        image_size=(256, 256),
        shuffle=True,
        seed=1024,
        validation_split=0.15,
        subset="both",
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False,
    )
    
    
    #load the test set
    test_set = tf.keras.utils.image_dataset_from_directory(
        directory=test_dir,
        labels='inferred',
        label_mode='int',
        class_names=class_names,
        color_mode='rgb',
        batch_size=32,
        image_size=(256, 256),
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False,
    )
    
    
    return [train_set, validation_set, test_set]
