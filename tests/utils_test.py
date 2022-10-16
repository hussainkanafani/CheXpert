import unittest
import numpy as np
import os
import util as util
import pandas as pd
from data_generator import DataGenerator


class UtilsTest(unittest.TestCase):
    test_image_path = "resources/test_image.jpg"
    mock_df = [['row1', 1], ['row2', 2], ['row3 ', 3],
               ['row4 ', 4], ['row5', 5], ['row6', 0]]

    def test_scale_image(self):
        path = self.test_image_path
        abspath = os.path.abspath("")
        path = os.path.join(abspath, path)
        img = util.preprocess_image(path)
        self.assertEqual(img.shape[:2], (320, 320))

    def test_read_existing_image(self,):
        path = self.test_image_path
        abspath = os.path.abspath("")
        # script_dir=os.path.dirname(__file__)
        path = os.path.join(abspath, path)
        img_array = util.preprocess_image(path)
        self.assertIsNotNone(img_array)

    def test_drop_rows_by_value(self):
        df = pd.DataFrame(self.mock_df, columns=['col1', 'col2'])
        self.assertEqual(len(df), 6)
        self.assertEqual(len(df[df['col2'] == 0]), 1)
        # drop rows with 0 value
        df = util.drop_rows_by_value(df, 0)
        self.assertEqual(len(df[df['col2'] == 0]), 0)

    def test_count_df_rows(self):
        labels = ['col1', 'col2']
        df = pd.DataFrame(self.mock_df, columns=labels)
        num_rows, _ = util.count_df_rows(df, labels)
        self.assertEqual(num_rows, 6)

    def test_creation_data_generator(self):
        dummy_csv = 'resources/dummy_valid.csv'
        class_names = ['Atelectasis', 'Cardiomegaly',
                       'Consolidation', 'Edema', 'Pleural Effusion']
        train_sequence = DataGenerator(
            csv_path=dummy_csv,
            class_names=class_names,
            batch_size=None,
            image_size=None,
            augmenter=None,
            steps=0,
            strategy=None)
        self.assertNotEquals(train_sequence, None)

    def test_falsy_file_path(self):
        class_names = ['Atelectasis', 'Cardiomegaly',
                       'Consolidation', 'Edema', 'Pleural Effusion']
        self.assertRaises(Exception, DataGenerator, "falsy.csv",
                          class_names, None, None, None, 0, None)

    def test_none_file_path(self):
        class_names = ['Atelectasis', 'Cardiomegaly',
                       'Consolidation', 'Edema', 'Pleural Effusion']        
        self.assertRaises(Exception, DataGenerator,None,class_names,None,None,None,0,None)        


if __name__ == '__main__':
    unittest.main()
