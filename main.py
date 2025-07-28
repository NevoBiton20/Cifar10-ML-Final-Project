
import numpy as np
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Preprocessing
from preprocessing.hog_features import extract_hog_features
from preprocessing.pca_reduction import apply_pca
from utils import flatten_images

# Models
from models.knn_model import train_knn
from models.svm_model import train_svm
from models.logistic_regression_model import train_logistic_regression
from models.random_forest_model import train_random_forest
from models.cnn_model import SimpleCNN

# Evaluation
from evaluation.evaluate_models import evaluate_model
from evaluation.confusion_matrix_analysis import plot_confusion_matrix

# Load dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X = np.concatenate([X_train, X_test])
y = np.concatenate([y_train, y_test]).flatten()

# Flatten and scale
X_flat = flatten_images(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_flat)

# --- Apply PCA ---
X_pca, _ = apply_pca(X_scaled, n_components=100)

# --- Apply HOG + PCA ---
X_hog = extract_hog_features(X)
X_hog_pca, _ = apply_pca(X_hog, n_components=100)

# Split for classical models
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)
X_train_hog, X_test_hog, y_train_hog, y_test_hog = train_test_split(X_hog_pca, y, test_size=0.2, random_state=42)

# --- Train ML Models on PCA ---
models_pca = {
    "KNN": train_knn(X_train_pca, y_train_pca),
    "SVM": train_svm(X_train_pca, y_train_pca),
    "LogReg": train_logistic_regression(X_train_pca, y_train_pca),
    "RF": train_random_forest(X_train_pca, y_train_pca)
}

print("\n--- Evaluation after PCA ---")
for name, model in models_pca.items():
    acc, report = evaluate_model(model, X_test_pca, y_test_pca)
    print(f"{name}: {acc:.4f}")

# --- Train ML Models on HOG + PCA ---
models_hog = {
    "KNN": train_knn(X_train_hog, y_train_hog),
    "SVM": train_svm(X_train_hog, y_train_hog),
    "LogReg": train_logistic_regression(X_train_hog, y_train_hog),
    "RF": train_random_forest(X_train_hog, y_train_hog)
}

print("\n--- Evaluation after HOG + PCA ---")
for name, model in models_hog.items():
    acc, report = evaluate_model(model, X_test_hog, y_test_hog)
    print(f"{name}: {acc:.4f}")

# --- CNN Pipeline (PyTorch) ---
def train_cnn(X, y, epochs=5, batch_size=64):
    X = torch.tensor(X / 255.0, dtype=torch.float32).permute(0, 3, 1, 2)
    y = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    print("\n--- CNN Evaluation ---")
    print(f"Accuracy: {acc:.4f}")
    print(report)

# Uncomment to train CNN
# train_cnn(X, y)
