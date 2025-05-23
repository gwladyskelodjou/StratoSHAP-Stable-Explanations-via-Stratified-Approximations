import numpy as np
import joblib
import os
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
import xgboost as xgb
from utils.data_loading import Data

# Seed for reproducibility
SEED = 42
np.random.seed(SEED)

datasets = ["bike", "parkinson", "german", "adult", "bank", "thyroid", "thoraric", "compas"]

models = {
    "SVM": {
        "classification": lambda: SVC(probability=True, random_state=SEED),
        "regression": lambda: SVR(C=100.0)
    },
    "MLP": {
        "classification": lambda: MLPClassifier(hidden_layer_sizes=(100,100), max_iter=1000, early_stopping=True, random_state=SEED),
        "regression": lambda: MLPRegressor(hidden_layer_sizes=(100,100), max_iter=1000, early_stopping=True, random_state=SEED)
    },
    "XGB": {
        "classification": lambda: xgb.XGBClassifier(random_state=SEED, eval_metric="logloss"),
        "regression": lambda: xgb.XGBRegressor(random_state=SEED)
    }
}

def train_model(model_builder, X_train, X_test, y_train, y_test, task_type):
    model = model_builder()
    model.fit(X_train.values, y_train)

    if task_type == "classification":
        score = model.score(X_test.values, y_test)
        print(f"Test accuracy: {score:.4f}")
    elif task_type == "regression":
        score = model.score(X_test.values, y_test)
        print(f"Test R² score: {score:.4f}")
    
    return model


def save_model(model, dataset_name, model_name, task_type, output_dir="Classifier"):
    os.makedirs(output_dir, exist_ok=True)

    dataset_dir = f"{output_dir}/{dataset_name}"
    os.makedirs(dataset_dir, exist_ok=True)
    
    filename = f"{dataset_name}/{dataset_name}_{model_name}.pkl"
    filepath = os.path.join(output_dir, filename)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")
    

for dataset in datasets:
    print(f"\n=== Dataset: {dataset} ===")
    data = Data(dataset_name=dataset)
    X_train, X_test, y_train, y_test = data.load_data()

    if dataset in ["bike", "parkinson"]:
        task_type = "regression"
    else:
        task_type = "classification"

    for model_name, model_variants in models.items():
        print(f"\nTraining {model_name} ({task_type})")
        model_builder = model_variants[task_type]
        model = train_model(model_builder, X_train, X_test, y_train, y_test, task_type)
        save_model(model, dataset, model_name, task_type)
