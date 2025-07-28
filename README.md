# CIFAR-10 Machine Learning Final Project

This project is the final deliverable for a university Machine Learning course.  
It focuses on applying **classical machine learning techniques** to the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) image classification dataset, along with a **CNN baseline** for performance comparison.

> 📌 This project avoids relying on deep learning as a primary method and instead emphasizes classical ML pipelines, feature extraction, dimensionality reduction, and evaluation.

---

## 🔍 Project Goals

- Classify images in the CIFAR-10 dataset using machine learning.
- Apply preprocessing steps: **HOG** (Histogram of Oriented Gradients) and **PCA** (Principal Component Analysis).
- Train and evaluate multiple ML models.
- Tune hyperparameters of each model.
- Compare results to a basic CNN baseline.

---

## 📁 Project Structure

```
cifar10-ml-final-project/
├── run_all.py               # Main entry point
├── requirements.txt         # Required Python packages
├── README.md                # This file
│
├── models/                  # Classical ML model training
│   └── train_classical_models.py
│
├── preprocess/              # Feature extraction modules
│   └── extract_features.py  # HOG + PCA
│
├── utils/                   # Evaluation and helper scripts
│   └── evaluation.py
│
├── report/                  # Results, plots, tuning logs (to be generated)
└── data/                    # (Optional) Place to store dataset manually
```

---

## 📦 Requirements

You can install all dependencies via:

```bash
pip install -r requirements.txt
```

### Main libraries used:
- `numpy`, `scikit-learn`, `matplotlib`, `pandas`
- `opencv-python` for HOG
- `torch`, `torchvision` for the CNN

---

## 🚀 How to Run the Project

After installing requirements:

```bash
python run_all.py
```

This will:
1. Run **HOG + PCA** on CIFAR-10 images
2. Train and evaluate:
   - Logistic Regression
   - K-Nearest Neighbors
   - Random Forest
   - Support Vector Machine
3. Train a **simple CNN baseline**
4. Output performance results and summary metrics

All results will be logged in the `report/` directory.

---

## 📉 Dimensionality Reduction

- **PCA** is used after HOG to reduce dimensionality.
- `n_components=100` was chosen (explained in the report).
- This helps models train faster and generalize better.

---

## 🧪 Models & Evaluation

Each classical model was evaluated using:
- **Accuracy**, **Precision**, **Recall**, **F1 Score**
- **Confusion Matrix** per class

Additionally:
- Each model was **fine-tuned** on its 2 most important hyperparameters.
- CNN was used **only as a baseline**, not as the project’s focus.

---

## 📊 Project Outputs

- Evaluation tables for all models (before/after PCA-HOG)
- Comparison to CNN
- Hyperparameter tuning logs
- Simulated full-dataset results (used due to resource limits)

---

## 💡 Notes

- This project simulates full CIFAR-10 results due to memory constraints.
- All optimizations (PCA, HOG) are applied together.
- CNN is not optimized — it serves as a lower-bound reference.

---

## ✍️ Report

The final report will include:
- Explanation of decisions, challenges, and optimizations
- Visualization of hyperparameter tuning
- Model performance summary table
- Discussion of results and insights per class

---

## 📚 Credits

- CIFAR-10 dataset: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
- Instructor: *Professor Lee-Ad Gottlieb / School of Computer Science*
- Student: **Nevo Biton**

---

## 📬 Questions?

Feel free to open an issue or contact [NevoBiton20](https://github.com/NevoBiton20).
