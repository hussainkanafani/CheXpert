import unittest
import numpy as np
import cv2
import data_augmentation.data_augmentation as imp
import os


class DataAugmentationTest(unittest.TestCase):
    test_image = "resources/test_image.jpg"

    def test_random_rotation(self):
        img_array = cv2.imread(self.test_image)
        self.assertIsNotNone(imp.random_rotation(img_array))
        self.assertNotEqual(imp.random_rotation(img_array).tolist(),img_array.tolist())

    def test_random_noise(self):
        img_array = cv2.imread(self.test_image)
        self.assertIsNotNone(imp.random_noise(img_array))
        self.assertNotEqual(imp.random_noise(img_array).tolist(),img_array.tolist())

    def test_horizontal_flip(self):
        img_array = cv2.imread(self.test_image)
        self.assertIsNotNone(imp.horizontal_flip(img_array))
        self.assertNotEqual(imp.horizontal_flip(img_array).tolist(),img_array.tolist())

    def test_vertical_flip(self):
        img_array = cv2.imread(self.test_image)
        self.assertIsNotNone(imp.vertical_flip(img_array))
        self.assertNotEqual(imp.vertical_flip(img_array).tolist(),img_array.tolist())

    def test_horizontal_shift(self):
        img_array = cv2.imread(self.test_image)
        self.assertIsNotNone(imp.horizontal_shift(img_array))
        self.assertNotEqual(imp.horizontal_shift(img_array).tolist(),img_array.tolist())

    def test_vertical_shift(self):
        img_array = cv2.imread(self.test_image)
        self.assertIsNotNone(imp.vertical_shift(img_array))
        self.assertNotEqual(imp.vertical_shift(img_array).tolist(),img_array.tolist())

    def test_random_brightness(self):
        img_array = cv2.imread(self.test_image)
        self.assertIsNotNone(imp.random_brightness(img_array))
        self.assertNotEqual(imp.random_brightness(img_array).tolist(),img_array.tolist())

    def test_random_zoomn(self):
        img_array = cv2.imread(self.test_image)
        self.assertIsNotNone(imp.random_zoom(img_array))
        self.assertNotEqual(imp.random_zoom(img_array).tolist(),img_array.tolist())


if __name__ == '__main__':
    unittest.main()
