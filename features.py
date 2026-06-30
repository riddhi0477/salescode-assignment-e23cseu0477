import cv2
import numpy as np
from skimage.feature import local_binary_pattern

TARGET_SIZE = 800

LBP_RADIUS = 2
LBP_POINTS = 8 * LBP_RADIUS
LBP_BINS = LBP_POINTS + 2  


def _load_and_resize(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    h, w = img.shape[:2]
    scale = TARGET_SIZE / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def extract_features(image_path: str) -> np.ndarray:
    img = _load_and_resize(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    feats = []

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    feats.append(lap.var())

    edges = cv2.Canny(gray, 100, 200)
    feats.append(edges.mean() / 255.0)

    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    hist, _ = np.histogram(lbp, bins=LBP_BINS, range=(0, LBP_BINS), density=True)
    feats.extend(hist.tolist())

    v = hsv[:, :, 2]
    feats.append(float((v > 245).mean()))
    feats.append(float((v < 10).mean()))

    s = hsv[:, :, 1].astype(np.float64)
    feats.append(s.mean() / 255.0)
    feats.append(s.std() / 255.0)

    b, g, r = cv2.split(img.astype(np.float64))
    feats.append(b.mean() / 255.0)
    feats.append(g.mean() / 255.0)
    feats.append(r.mean() / 255.0)
    feats.append((b.mean() - r.mean()) / 255.0) 

    block = 32
    hh, ww = gray.shape
    local_stds = []
    for y in range(0, hh - block, block):
        for x in range(0, ww - block, block):
            local_stds.append(gray[y:y + block, x:x + block].std())
    local_stds = np.array(local_stds) if local_stds else np.array([0.0])
    feats.append(local_stds.mean())
    feats.append(local_stds.std())

    feats.append(gray.mean() / 255.0)
    feats.append(gray.std() / 255.0)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=120,
                             minLineLength=int(0.3 * max(gray.shape)), maxLineGap=10)
    n_lines = 0 if lines is None else len(lines)
    feats.append(float(n_lines))

    h = hsv[:, :, 0]
    hue_hist, _ = np.histogram(h, bins=32, range=(0, 180), density=True)
    hue_hist = hue_hist + 1e-9
    hue_entropy = -np.sum(hue_hist * np.log(hue_hist))
    feats.append(float(hue_entropy))

    return np.array(feats, dtype=np.float64)


FEATURE_NAMES = (
    ["laplacian_var", "edge_density"]
    + [f"lbp_bin_{i}" for i in range(LBP_BINS)]
    + [
        "glare_frac",
        "deep_black_frac",
        "sat_mean",
        "sat_std",
        "b_mean",
        "g_mean",
        "r_mean",
        "blue_red_cast",
        "local_contrast_mean",
        "local_contrast_std",
        "brightness",
        "contrast",
        "hough_line_count",
        "hue_entropy",
    ]
)
