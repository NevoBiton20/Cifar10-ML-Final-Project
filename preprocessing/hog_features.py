
from skimage.feature import hog
import numpy as np

def extract_hog_features(images):
    hog_features = []
    for image in images:
        feature = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), multichannel=True)
        hog_features.append(feature)
    return np.array(hog_features)
