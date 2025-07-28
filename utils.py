
import numpy as np
import matplotlib.pyplot as plt

def flatten_images(images):
    return images.reshape(images.shape[0], -1)

def plot_class_distribution(y, class_names, save_path="class_distribution.png"):
    counts = np.bincount(y)
    plt.bar(class_names, counts)
    plt.xticks(rotation=45)
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
