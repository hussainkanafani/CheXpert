import numpy as np
import pandas as pd
import cv2
import sys
import getopt
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
from keras.applications.densenet import preprocess_input
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam, SGD
from keras.models import Model, load_model
from keras.layers import *
from model import ModelFactory
import keras
from keras.callbacks import *
import util
from data_generator import DataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from augmenter import flip_augmenter
from augmenter import complex_augmenter


def main(epochs, weights_path, batch_size, data_augmenter=None):
    #class_names = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema','Pleural Effusion']
    class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                   'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
                   'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
    model_name = "DenseNet121"
    csv_train_path = "CheXpert-v1.0-small/train.csv"
    csv_valid_path = "CheXpert-v1.0-small/valid.csv"
    train_df = pd.read_csv(csv_train_path)
    valid_df = pd.read_csv(csv_valid_path)

    preprocessing_strategy = "ignore"
    if(preprocessing_strategy == "ignore"):
        train_df=util.drop_rows_by_value(train_df,-1)

    train_counts, train_pos_counts = util.count_df_rows(train_df,class_names)
    dev_counts, _ = util.count_df_rows(valid_df, class_names)
    c_augmenter = data_augmenter
    initial_learning_rate = 0.0001
    generator_workers = 1
    image_dimension = 320
    train_steps = int(train_counts / batch_size)
    patience_reduce_lr = 1
    min_lr = 1e-8
    validation_steps = int(dev_counts / batch_size)

    train_sequence = DataGenerator(
        csv_path=csv_train_path,
        class_names=class_names,
        batch_size=batch_size,
        image_size=(image_dimension, image_dimension),
        augmenter=c_augmenter,
        steps=train_steps,
        strategy=preprocessing_strategy)

    validation_sequence = DataGenerator(
        csv_path=csv_valid_path,
        class_names=class_names,
        batch_size=batch_size,
        image_size=(image_dimension, image_dimension),
        augmenter=None,
        steps=validation_steps,
        strategy=preprocessing_strategy)

    model_factory = ModelFactory()
    model = model_factory.get_model(
        class_names,
        model_name=model_name,
        weights_path=weights_path,
        input_shape=(image_dimension, image_dimension, 3))

    output_weights_path = "Model.h5"
    model_train = model
    checkpoint = ModelCheckpoint(
        output_weights_path,
        save_weights_only=False,
        save_best_only=True,
        verbose=1,
        period=1)
    model.summary()

    optimizer = Adam(lr=initial_learning_rate)
    model_train.compile(
        metrics=['accuracy'], optimizer=optimizer, loss="binary_crossentropy")

    callbacks = [
        checkpoint,
        TensorBoard(log_dir=os.path.join("", "logs"), batch_size=batch_size),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience_reduce_lr,
                          verbose=1, mode="min", min_lr=min_lr),
    ]

    history = model_train.fit_generator(
        generator=train_sequence,
        steps_per_epoch=train_steps,
        epochs=epochs,
        validation_data=validation_sequence,
        validation_steps=validation_steps,
        callbacks=callbacks,
        class_weight=None,
        workers=8,
        shuffle=False)


if __name__ == '__main__':
    argv = sys.argv[1:]
    epochs = 3
    weights_path = None
    batch_size = 32
    data_augmenter = None
    try:
        opts, args = getopt.getopt(argv,"a:b:e:w:")
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-b","--batchSize"):  
            batch_size = int(arg)
        elif opt in ("-w","--weightsPath"):
            weights_path = arg
        elif opt in ("-e","--epochs"): 
            epochs = int(arg) 
        elif opt in ("-a","--augmentation"):
            if arg=='flip':
                data_augmenter = flip_augmenter
            elif arg=='complex':
                data_augmenter = complex_augmenter
            else:
                data_augmenter = None	
    main(epochs, weights_path, batch_size,data_augmenter)
