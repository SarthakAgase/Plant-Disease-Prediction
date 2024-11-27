import unittest
import numpy as np
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model # type: ignore
from tqdm import tqdm
import time


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
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42 ,shuffle=True)


class TestPotatoDiseaseModelIntegration(unittest.TestCase):

    def test_train_test_split(self):
        """Test: If the training and testing data are split with the correct proportions."""
        total = len(images)
        train_ratio = len(X_train) / total
        test_ratio = len(X_test) / total
        self.assertAlmostEqual(train_ratio, 0.8, delta=0.05, msg="Train-test split ratio is incorrect.")
        self.assertAlmostEqual(test_ratio, 0.2, delta=0.05, msg="Train-test split ratio is incorrect.")

    def test_model_training(self):
        """Test: If the model is able to run through an epoch without errors."""
        history = Model.fit(X_train, y_train, epochs=1, batch_size=100, verbose=0)
        self.assertIn('accuracy', history.history, "Training accuracy not found in history.")
        self.assertIn('loss', history.history, "Training loss not found in history.")

    def test_model_predictions(self):
        """Test: If the model produces predictions of expected shape."""
        y_pred = Model.predict(X_test)
        self.assertEqual(y_pred.shape[0], len(X_test), "Number of predictions does not match test set size.")
        self.assertEqual(y_pred.shape[1], 3, "Predictions do not have correct number of classes.")

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
    runner.run(unittest.makeSuite(TestPotatoDiseaseModelIntegration))
