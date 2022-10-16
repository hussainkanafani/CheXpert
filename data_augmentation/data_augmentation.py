import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
import numpy as np
import matplotlib.pyplot as plt
import cv2
from numpy import expand_dims
from keras.preprocessing.image import ImageDataGenerator


def random_rotation(image_array):
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)


def random_noise(image_array):
    return sk.util.random_noise(image_array)


def horizontal_flip(image_array):
    return image_array[:, ::-1]


def vertical_flip(image_array: ndarray):
    return image_array[::-1, :]


def horizontal_shift(image_array):
    samples = expand_dims(image_array, 0)
    datagen = ImageDataGenerator(width_shift_range=[-100, 100])
    it = datagen.flow(samples, batch_size=1)
    image = it.next()[0].astype('uint32')
    return image


def vertical_shift(image_array):
    samples = expand_dims(image_array, 0)
    datagen = ImageDataGenerator(height_shift_range=[-100, 100])
    it = datagen.flow(samples, batch_size=1)
    image = it.next()[0].astype('uint32')
    return image


def random_brightness(image_array):
    samples = expand_dims(image_array, 0)
    datagen = ImageDataGenerator(brightness_range=[0.2, 1.0])
    it = datagen.flow(samples, batch_size=1)
    image = it.next()[0].astype('uint32')
    return image


def random_zoom(image_array):
    samples = expand_dims(image_array, 0)
    datagen = ImageDataGenerator(zoom_range=[0.5, 1.0])
    it = datagen.flow(samples, batch_size=1)
    image = it.next()[0].astype('uint32')
    return image
