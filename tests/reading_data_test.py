import unittest
import numpy as np
from prediction import load_image
from prediction import load_studies

class ReadingDataTest(unittest.TestCase):
    test_image = "http://scikit-image.org/_static/img/logo.png"

    def test_load_image(self):
        image_320 = load_image('resources/test_image.jpg',320)
        self.assertEqual(image_320.shape,(1,320,320,3))
        image_224 = load_image('resources/test_image.jpg',224)
        self.assertEqual(image_224.shape,(1,224,224,3))

    def test_load_study_(self):
        dummy_csv = 'resources/dummy_valid.csv'
        # dummy csv contains 4 entries, 3 studies. 1 study doesn't point to an actual image, 1 has 2 images
        # expected: 2 valid studies (containing images), first study should have 2 images, second should have 1 image
        studies = load_studies(dummy_csv)
        self.assertEqual(len(studies),2)
        self.assertEqual(len(studies[0]),3) # 2 images + name of the study
        self.assertEqual(len(studies[1]),2)



if __name__ == '__main__':
    unittest.main()