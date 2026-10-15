"""
NutriScan - Ingredient Category Classifier
Trains and compares SVM, Logistic Regression, and KNN on TF-IDF features
extracted from ingredient names, to pick the best-performing model.
"""

import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, classification_report
)

DATA_PATH = "data/ingredient_knowledge_base.csv"
MODEL_DIR = "models"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["ingredient_name", "category"])
    return df


def build_features(df):
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    X = vectorizer.fit_transform(df["ingredient_name"])
    y = df["category"]
    return X, y, vectorizer


def evaluate_model(name, model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted", zero_division=0)
    rec = recall_score(y_test, preds, average="weighted", zero_division=0)
    f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
    print(f"\n--- {name} ---")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1-score:  {f1:.3f}")
    return {"name": name, "model": model, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def main():
    df = load_data()
    print(f"Loaded {len(df)} labeled ingredients across {df['category'].nunique()} categories.")
    print(df["category"].value_counts())

    X, y, vectorizer = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    results = []
    results.append(evaluate_model("SVM", SVC(kernel="linear", C=1.0).fit(X_train, y_train), X_test, y_test))
    results.append(evaluate_model("Logistic Regression", LogisticRegression(max_iter=1000).fit(X_train, y_train), X_test, y_test))
    results.append(evaluate_model("KNN (k=3)", KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train), X_test, y_test))

    best = max(results, key=lambda r: r["f1"])
    print(f"\n=== Best model: {best['name']} (F1 = {best['f1']:.3f}) ===")
    print("\nClassification report for best model:")
    print(classification_report(y_test, best["model"].predict(X_test), zero_division=0))

    joblib.dump(best["model"], f"{MODEL_DIR}/ingredient_classifier.pkl")
    joblib.dump(vectorizer, f"{MODEL_DIR}/tfidf_vectorizer.pkl")
    print(f"\nSaved best model ({best['name']}) and vectorizer to {MODEL_DIR}/")

    comparison_df = pd.DataFrame([{k: v for k, v in r.items() if k != "model"} for r in results])
    comparison_df.to_csv(f"{MODEL_DIR}/model_comparison.csv", index=False)
    print(f"Saved model comparison table to {MODEL_DIR}/model_comparison.csv")


if __name__ == "__main__":
    main()
