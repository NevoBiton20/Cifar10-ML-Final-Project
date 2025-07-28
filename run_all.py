# Orchestrator for the entire ML pipeline
from models.train_classical_models import train_and_evaluate_all
from models.train_cnn import run_cnn_pipeline
from preprocess.extract_features import extract_hog_pca
from utils.evaluation import summarize_results

if __name__ == '__main__':
    print("Starting CIFAR-10 ML project pipeline...")
    extract_hog_pca()
    train_and_evaluate_all()
    run_cnn_pipeline()
    summarize_results()
