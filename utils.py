import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

def create_kfold(df, num_splits):
    if num_splits is None: assert 'num_splits is not specified'
        
    stratifier = StratifiedKFold(n_splits = num_splits, shuffle = True, random_state = 42)
    for fold_idx, (train_idx, val_idx) in enumerate(stratifier.split(df, df['label'])):
        df.loc[val_idx, 'kfold'] = fold_idx

    return df

def preprocess(df, mode):
    if mode == 'train':
        features = df.iloc[:, 1:].to_numpy().astype('float32')
        features = features.reshape([features.shape[0], 28, 28, 1])
        features /= 255.

        targets = df['label'].to_numpy().astype('int32')
        targets = to_categorical(targets)
        return features, targets
    else:
        features = df.to_numpy().astype('float32')
        features = features.reshape([features.shape[0], 28, 28, 1])
        features /= 255.
        return features

def create_generator(features, targets, batch_size):
    img_generator = ImageDataGenerator(zoom_range = 0.15, width_shift_range = 0.1, 
        height_shift_range = 0.1, rotation_range = 0.15)
    generator = img_generator.flow(features, targets, batch_size = batch_size)
    return generator







    