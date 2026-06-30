import argparse
import glob
import os
import pickle
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

from features import extract_features

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def list_images(folder):
    files = set()
    for ext in IMG_EXTS:
        for f in glob.glob(os.path.join(folder, ext)):
            files.add(os.path.normcase(os.path.abspath(f)))
    return sorted(files)


def build_dataset(data_dir):
    real_files = list_images(os.path.join(data_dir, "real"))
    screen_files = list_images(os.path.join(data_dir, "screen"))
    print(f"Found {len(real_files)} real photos, {len(screen_files)} screen photos")

    X, y, paths = [], [], []
    t0 = time.time()
    for f in real_files:
        X.append(extract_features(f))
        y.append(0)
        paths.append(f)
    for f in screen_files:
        X.append(extract_features(f))
        y.append(1)
        paths.append(f)
    elapsed = time.time() - t0
    n = len(X)
    print(f"Feature extraction: {elapsed:.2f}s total, {1000 * elapsed / n:.1f} ms/image avg")

    return np.array(X), np.array(y), paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="dataset", help="folder containing real/ and screen/ subfolders")
    args = ap.parse_args()

    X, y, paths = build_dataset(args.data_dir)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")

    cv_preds = cross_val_predict(clf, Xs, y, cv=skf, method="predict")
    cv_proba = cross_val_predict(clf, Xs, y, cv=skf, method="predict_proba")[:, 1]

    acc = accuracy_score(y, cv_preds)
    prec = precision_score(y, cv_preds)
    rec = recall_score(y, cv_preds)
    f1 = f1_score(y, cv_preds)
    auc = roc_auc_score(y, cv_proba)
    cm = confusion_matrix(y, cv_preds)

    print("\n===== 5-fold cross-validation results (100 photos, 50/50 split) =====")
    print(f"Accuracy:  {acc * 100:.1f}%")
    print(f"Precision: {prec * 100:.1f}%")
    print(f"Recall:    {rec * 100:.1f}%")
    print(f"F1 score:  {f1 * 100:.1f}%")
    print(f"ROC AUC:   {auc:.3f}")
    print("Confusion matrix (rows=true, cols=pred) [real, screen]:")
    print(cm)

    misclassified = [paths[i] for i in range(len(y)) if cv_preds[i] != y[i]]
    if misclassified:
        print(f"\nMisclassified ({len(misclassified)}):")
        for m in misclassified:
            print(f"  {m}")

    Xtr, Xte, ytr, yte, ptr, pte = train_test_split(
        X, y, paths, test_size=0.2, stratify=y, random_state=0
    )
    scaler2 = StandardScaler().fit(Xtr)
    clf2 = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf2.fit(scaler2.transform(Xtr), ytr)
    test_acc = clf2.score(scaler2.transform(Xte), yte)
    print(f"\nHeld-out 80/20 split accuracy: {test_acc * 100:.1f}% (n_test={len(yte)})")

    sample = paths[0]
    n_reps = 20
    clf_final = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf_final.fit(Xs, y)
    t0 = time.time()
    for _ in range(n_reps):
        f = extract_features(sample)
        fs = scaler.transform([f])
        clf_final.predict_proba(fs)
    elapsed = (time.time() - t0) / n_reps
    print(f"\nPer-image latency (feature extraction + inference): {elapsed * 1000:.1f} ms "
          f"(CPU, averaged over {n_reps} runs on '{os.path.basename(sample)}')")
    print("Cost per image: $0 (runs fully on-device / CPU, no API calls)")

    final_scaler = StandardScaler().fit(X)
    final_clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    final_clf.fit(final_scaler.transform(X), y)

    with open("model.pkl", "wb") as f:
        pickle.dump({"scaler": final_scaler, "clf": final_clf}, f)
    print("\nSaved trained model to model.pkl")


if __name__ == "__main__":
    main()