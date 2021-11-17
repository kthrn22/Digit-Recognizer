import numpy as np
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, MaxPool2D, BatchNormalization, Dropout, Conv2D
from tensorflow.keras.optimizers import Adam
from utils import *
from modeling import *
from keras import callbacks
from keras.callbacks import LearningRateScheduler, EarlyStopping

class Model():
    def __init__(self, dropout_probs, hidden_size):
        super(Model, self).__init__()
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), padding = 'same', activation = 'relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(32, (3, 3), padding = 'same', activation = 'relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(32, (5, 5), padding = 'same', activation = 'relu', strides = 2))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.4))
    
        self.model.add(Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(64, (3, 3), padding = 'same', activation = 'relu', strides = 2))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(dropout_probs))
    
        self.model.add(Flatten())
        self.model.add(Dense(hidden_size, activation = 'relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dense(10, activation = 'softmax'))
        self.model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    def train(self, train_data, validation_data, patience, epochs, batch_size):
        X_train, y_train = train_data
        X_val, y_val = validation_data

        train_generator = create_generator(X_train, y_train, batch_size)

        annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
        early_stop = EarlyStopping(monitor = 'val_loss', patience = patience)

        self.model.fit(train_generator, validation_data = validation_data, epochs = epochs, 
                steps_per_epoch = X_train.shape[0] // batch_size, callbacks = [annealer, early_stop])

    def predict(self, features, predict_probs):
        probs = self.model.predict(features)
        labels = np.argmax(probs, axis = -1)
        if predict_probs:
            return labels, probs
        return labels