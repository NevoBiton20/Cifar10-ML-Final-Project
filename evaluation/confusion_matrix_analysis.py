
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def plot_confusion_matrix(model, X_test, y_test, class_names, class_index):
    disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=class_names)
    plt.title(f'Confusion Matrix for Class {class_index}')
    plt.savefig(f'confusion_matrix_knn_class{class_index}.png')
    plt.close()
