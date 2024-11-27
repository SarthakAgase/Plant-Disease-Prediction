import unittest
import numpy as np
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model # type: ignore
from tqdm import tqdm
from sklearn.metrics import classification_report
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

class TestPotatoDiseaseModelAcceptance(unittest.TestCase):

    def test_final_accuracy(self):
        """Test: If the final model accuracy meets the expected threshold."""
        _, accuracy = Model.evaluate(X_test, y_test, verbose=0)
        self.assertGreaterEqual(accuracy, 0.80, "Model accuracy is below acceptable threshold of 80%.")

    def test_classification_report(self):
        """Test: If classification report is generated and all classes are covered."""
        test_result = [np.argmax(x) for x in Model.predict(X_test)]
        report = classification_report(y_test, test_result, output_dict=True)
        for label in ['0', '1', '2']:
            self.assertIn(label, report, f"Classification report missing label {label}")

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
    runner.run(unittest.makeSuite(TestPotatoDiseaseModelAcceptance))
