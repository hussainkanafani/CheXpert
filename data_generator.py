import numpy as np
import os
from keras.utils import Sequence
import pandas as pd
import cv2
import util


class DataGenerator(Sequence):
    def __init__(self, csv_path, class_names, batch_size=16,
                 image_size=(320, 320), augmenter=None, verbose=0, steps=None, strategy="ignore"):
        
        
        if(not csv_path or not os.path.isfile(csv_path)):
            raise(Exception)
        self.df = pd.read_csv(csv_path)        
        self.batch_size = batch_size
        self.image_size = image_size
        self.augmenter = augmenter
        self.verbose = verbose
        self.class_names = class_names
        self.strategy = strategy
        self.preprocess_data()
        if steps is None:
            self.steps = int(
                np.ceil(len(self.x_path) / float(self.batch_size)))
        else:
            self.steps = int(steps)

    def __bool__(self):
        return True

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        batch_x_path = self.x_path[idx *
                                   self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.asarray([self.load_image_and_resize(x_path)
                              for x_path in batch_x_path])
        batch_x = self.standardize_images(batch_x)
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

    def load_image_and_resize(self, path):
        img = util.preprocess_image(path)
        return img

    def standardize_images(self, img_batch):
        if self.augmenter is not None:
            img_batch = self.augmenter.augment_images(img_batch)
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        # mean equal zero with std of 1
        img_batch = (img_batch - imagenet_mean) / imagenet_std
        return img_batch

    def get_y_true(self):
        return self.y[:self.steps*self.batch_size, :]

    def preprocess_data(self):
        df = self.df.sample(frac=1., random_state=1)
        # fill nan values
        df.fillna(0, inplace=True)
        if (self.strategy == "ones"):
            df.replace(-1, 1, inplace=True)
        elif (self.strategy == "zeroes"):
            df.replace(-1, 0, inplace=True)
        elif (self.strategy == "ignore"):
            df = util.drop_rows_by_value(df, -1)
        self.x_path, self.y = df["Path"].values, df[self.class_names].values
