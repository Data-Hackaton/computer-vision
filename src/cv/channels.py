"""9 каналов на вход детектора: исходный кадр | резкость | контраст (BGR каждого). Проверка, поможет ли детектору
мелочь вроде купюр в руке; одна функция stack9 — и для датасета, и для инференса.

  python -m cv.channels --src det_l1f --dst det_l1f9   → data/datasets/det_l1f9/yolo: 9-канальные .tiff, те же метки,
                                                          data.yaml с channels: 9 (Ultralytics читает страницы TIFF
                                                          одного размера как каналы)
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

from cv.detectors import DATA_DIR

SHARPEN_SIGMA = 2.0
CLAHE = dict(clipLimit=2.0, tileGridSize=(8, 8))


def sharpen(bgr: np.ndarray) -> np.ndarray:
    """Нерезкое маскирование: 1,5 × кадр − 0,5 × размытый."""
    return cv2.addWeighted(bgr, 1.5, cv2.GaussianBlur(bgr, (0, 0), SHARPEN_SIGMA), -0.5, 0)


def contrast(bgr: np.ndarray) -> np.ndarray:
    """CLAHE по яркости (L в LAB), цвет не трогаем."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(**CLAHE).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def stack9(bgr: np.ndarray) -> np.ndarray:
    return np.concatenate([bgr, sharpen(bgr), contrast(bgr)], axis=2)


def write_tiff(path: Path, img: np.ndarray) -> None:
    ok = cv2.imwritemulti(str(path), [np.ascontiguousarray(img[..., c]) for c in range(img.shape[2])])
    if not ok:
        raise RuntimeError(f"не записан {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="det_l1f")
    ap.add_argument("--dst", default="det_l1f9")
    args = ap.parse_args()
    src, dst = DATA_DIR / "datasets" / args.src / "yolo", DATA_DIR / "datasets" / args.dst / "yolo"
    if dst.parent.exists():
        shutil.rmtree(dst.parent)
    n = 0
    for split in ("train", "val"):
        (dst / "images" / split).mkdir(parents=True)
        shutil.copytree(src / "labels" / split, dst / "labels" / split)
        for jpg in sorted((src / "images" / split).glob("*.jpg")):
            write_tiff(dst / "images" / split / f"{jpg.stem}.tiff", stack9(cv2.imread(str(jpg))))
            n += 1
    names = (src / "data.yaml").read_text().split("names:", 1)[1]
    (dst / "data.yaml").write_text(f"path: {dst}\ntrain: images/train\nval: images/val\nchannels: 9\nnames:{names}")
    print(f"→ {dst}: {n} кадров по 9 каналов")


if __name__ == "__main__":
    main()
