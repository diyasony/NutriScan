"""
NutriScan - Ingredient Category Classifier

Trains and compares LinearSVC (documented main model), Logistic Regression, KNN, Decision Tree
and Random Forest on TF-IDF char_wb (2,4) features of ingredient names.

Evaluation uses GroupKFold(5) grouped by parent_ingredient, so a name and its aliases / OCR-noise
variants never sit in both train and test (otherwise scores would be inflated by near-duplicates).
The TF-IDF vectoriser is re-fitted inside every training fold.

Run from the project folder:  python train_classifier.py
"""

import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "nutriscan_training_variants_v2.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["ingredient_name", "category", "parent_ingredient"])
    df["ingredient_name"] = df["ingredient_name"].astype(str).str.lower().str.strip()
    return df.reset_index(drop=True)


def new_vectorizer():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))


def make_models():
    return {
        "LinearSVC (SVM)": LinearSVC(C=1.0, random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "KNN (k=3)": KNeighborsClassifier(n_neighbors=3),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    }


def make_splitter():
    try:
        return GroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    except TypeError:                       # older scikit-learn: no shuffle argument
        return GroupKFold(n_splits=5)


def cross_validate_all(df):
    texts, y, groups = df["ingredient_name"].values, df["category"].values, df["parent_ingredient"].values
    splits = list(make_splitter().split(texts, y, groups))
    rows, oof = [], {}
    for name, model in make_models().items():
        pooled = np.empty(len(y), dtype=object)
        fold_scores = {"accuracy": [], "precision": [], "recall": [], "f1": []}
        for tr, te in splits:
            vec = new_vectorizer()
            Xtr = vec.fit_transform(texts[tr])
            clf = make_models()[name].fit(Xtr, y[tr])
            pred = clf.predict(vec.transform(texts[te]))
            pooled[te] = pred
            fold_scores["accuracy"].append(accuracy_score(y[te], pred))
            fold_scores["precision"].append(precision_score(y[te], pred, average="weighted", zero_division=0))
            fold_scores["recall"].append(recall_score(y[te], pred, average="weighted", zero_division=0))
            fold_scores["f1"].append(f1_score(y[te], pred, average="weighted", zero_division=0))
        row = {"model": name}
        for k, v in fold_scores.items():
            row[k] = round(float(np.mean(v)), 4)
        row["f1_std_over_folds"] = round(float(np.std(fold_scores["f1"])), 4)
        rows.append(row)
        oof[name] = pooled
        print(f"{name:22s} acc={row['accuracy']:.3f} prec={row['precision']:.3f} "
              f"rec={row['recall']:.3f} f1={row['f1']:.3f} (+/-{row['f1_std_over_folds']:.3f})")
    return pd.DataFrame(rows), oof


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_data()
    print(f"Loaded {len(df)} rows, {df['parent_ingredient'].nunique()} parent ingredients, "
          f"{df['category'].nunique()} categories.")
    print(df["category"].value_counts().to_string())
    print("\nGroupKFold(5) grouped by parent_ingredient (mean over folds):")

    comparison, oof = cross_validate_all(df)
    comparison = comparison.sort_values("f1", ascending=False).reset_index(drop=True)
    comparison.to_csv(os.path.join(MODEL_DIR, "model_comparison.csv"), index=False)

    best_name = comparison.loc[0, "model"]
    print(f"\n=== Best model by weighted F1: {best_name} (F1 = {comparison.loc[0, 'f1']:.3f}) ===")
    if best_name != "LinearSVC (SVM)":
        svm_f1 = float(comparison.loc[comparison.model == "LinearSVC (SVM)", "f1"].iloc[0])
        print(f"NOTE: the documented main model (LinearSVC) did NOT win: F1 {svm_f1:.3f} vs "
              f"{comparison.loc[0, 'f1']:.3f}. The saved model is {best_name}.")

    y = df["category"].values
    pred = oof[best_name].astype(str)
    report = classification_report(y, pred, zero_division=0)
    with open(os.path.join(MODEL_DIR, "classification_report.txt"), "w") as f:
        f.write(f"Best model: {best_name}\n"
                f"Out-of-fold predictions, GroupKFold(5) by parent_ingredient, {len(y)} rows\n\n{report}")
    print(report)

    labels = sorted(set(y))
    cm = confusion_matrix(y, pred, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(f"Confusion matrix - {best_name} (out-of-fold, grouped CV)")
    fig.tight_layout()
    fig.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=150)

    # final model: fit on ALL rows; app/pipeline use vectorizer.transform -> classifier.predict
    vec = new_vectorizer()
    X = vec.fit_transform(df["ingredient_name"])
    final = make_models()[best_name].fit(X, y)
    joblib.dump(final, os.path.join(MODEL_DIR, "ingredient_classifier.pkl"))
    joblib.dump(vec, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    print(f"\nSaved {best_name} + vectorizer to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
