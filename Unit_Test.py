import unittest
import numpy as np
import pandas as pd
import os
import cv2
from tensorflow.keras.models import load_model # type: ignore
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

Model = load_model("src\Potato.keras")

dataDir = 'datasets\Potato'
selectedClasses = ['Potato___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   ]

imgPaths = []
labels = []
for className in os.listdir(dataDir):
    if className in selectedClasses:
        classPath = os.path.join(dataDir, className)
        for img in os.listdir(classPath):
            imgPath = os.path.join(classPath, img)
            imgPaths.append(imgPath)
            labels.append(className)

df = pd.DataFrame({
    'imgPath': imgPaths,
    'label': labels
})

df['label'] = df['label'].replace({'Potato___healthy': 0,
                                   'Potato___Early_blight': 1,
                                   'Potato___Late_blight': 2,
                                   }).astype(int)

IMG_SIZE = (150, 150)
imgs = []
for imgPath in tqdm(df['imgPath'], total=len(df)):
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    imgs.append(img)

images = np.array(imgs)
labels = np.array(df['label'])

images = images / 255.0


class TestPotatoDiseaseModel(unittest.TestCase):
    def test_data_loading(self):
        """Test: If images and labels are loaded and have matching dimensions."""
        self.assertEqual(len(images), len(labels),
                         "Images and labels count mismatch.")
        self.assertGreater(len(images), 0, "No images loaded.")
        self.assertGreater(len(labels), 0, "No labels loaded.")

    def test_image_preprocessing(self):
        """Test: If images are correctly preprocessed to the target size and normalized."""
        self.assertEqual(
            images.shape[1:], (150, 150, 3), "Images are not of the expected shape.")
        self.assertTrue(np.all(images <= 1) and np.all(
            images >= 0), "Images are not normalized correctly.")

    def test_model_compilation(self):
        """Test: If the model is compiled with the correct parameters."""
        self.assertEqual(
            Model.loss, 'sparse_categorical_crossentropy', "Loss function is incorrect.")
        self.assertEqual(Model.optimizer.name, 'adam',
                         "Optimizer is not Adam.")
        self.assertIn(
            'compile_metrics', [m.name for m in Model.metrics], "Compile metric is missing.")

    def test_model_input_output_shape(self):
        """Test: If the model input and output shapes are as expected."""
        self.assertEqual(Model.input_shape, (None, 150, 150, 3),
                         "Input shape is incorrect.")
        self.assertEqual(Model.output_shape, (None, 3),
                         "Output shape is incorrect.")

class MyTestResult(unittest.TestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = time.time()

    def startTest(self, test):
        super().startTest(test)
        self.test_start_time = time.time()

    def stopTest(self, test):
        super().stopTest(test)
        test_time = time.time() - self.test_start_time
        print(f"{test.shortDescription()} took {test_time:.2f} seconds")

if __name__ == '__main__':
    # unittest.main()
    runner = unittest.TextTestRunner(resultclass=MyTestResult)
    runner.run(unittest.makeSuite(TestPotatoDiseaseModel))
