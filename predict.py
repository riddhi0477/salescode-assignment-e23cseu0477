import os
import pickle
import sys

from features import extract_features

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")


def predict(image_path: str) -> float:
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    scaler, clf = bundle["scaler"], bundle["clf"]

    feats = extract_features(image_path).reshape(1, -1)
    feats_scaled = scaler.transform(feats)
    score = clf.predict_proba(feats_scaled)[0, 1]
    return float(score)


if __name__ == "__main__":
    print(predict(sys.argv[1]))
