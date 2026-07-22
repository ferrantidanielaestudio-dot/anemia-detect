"""
Pre-inference image validation checks.

This module holds lightweight, classical-CV gatekeeping checks that run
BEFORE the image reaches the anemia classification model. It is kept
separate from app.py so new checks can be added here later without
touching the FastAPI routing logic.

Checks implemented so far:
  - detect_eye(): is there a human eye in the frame at all?
  - detect_eye() also reports eyes_open: is the eye actually open (pupil
    visible), since a closed eye hides the conjunctiva the model needs.
"""

import os
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

# cv2 ships pretrained Haar cascade classifiers (Viola-Jones detectors,
# trained on thousands of labeled eye/face images) inside the package
# itself, so no extra model download is required at build or run time.
_CASCADE_DIR = cv2.data.haarcascades

_eye_cascade = cv2.CascadeClassifier(os.path.join(_CASCADE_DIR, "haarcascade_eye.xml"))
_face_cascade = cv2.CascadeClassifier(
    os.path.join(_CASCADE_DIR, "haarcascade_frontalface_default.xml")
)

if _eye_cascade.empty() or _face_cascade.empty():
    raise RuntimeError(
        "No se pudieron cargar los clasificadores Haar de OpenCV. "
        "Verifica la instalación de opencv-python-headless."
    )


@dataclass
class EyeDetectionResult:
    eye_found: bool
    eyes_detected: int
    face_detected: bool
    eyes_open: bool
    eye_boxes: list = field(default_factory=list)  # [(x, y, w, h), ...] in `gray` coordinates


def _to_gray_array(image: Image.Image) -> np.ndarray:
    gray_image = image.convert("L")
    return np.array(gray_image)


def _has_visible_pupil(eye_roi: np.ndarray) -> bool:
    """
    Confirms an *open* eye by looking for the pupil/iris: a small, dark,
    roughly circular blob against the lighter sclera. A closed eyelid has
    no such structure (just skin/eyelash texture), so this reliably tells
    open from closed without any color rules — it looks at shape (via
    Hough circle transform) and local contrast, not hue.

    eye_roi: grayscale crop of a single detected eye region.
    """
    h, w = eye_roi.shape[:2]
    if h < 10 or w < 10:
        return False

    blurred = cv2.GaussianBlur(eye_roi, (5, 5), 0)

    min_radius = max(2, int(min(h, w) * 0.12))
    max_radius = max(min_radius + 1, int(min(h, w) * 0.45))

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(h, w),
        param1=80,
        param2=22,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return False

    # A real pupil is noticeably darker than the eye region around it.
    # This rejects circles Hough finds on eyelid folds/eyelashes in a
    # closed eye, where there is no strong dark blob at all.
    roi_mean = float(np.mean(eye_roi))
    for cx, cy, r in circles[0]:
        cx, cy, r = int(cx), int(cy), max(1, int(r * 0.6))
        y0, y1 = max(0, cy - r), min(h, cy + r)
        x0, x1 = max(0, cx - r), min(w, cx + r)
        patch = eye_roi[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        if float(np.mean(patch)) < roi_mean - 15:
            return True

    return False


def detect_eye(image: Image.Image) -> EyeDetectionResult:
    """
    Determines whether a human eye is present in the image, and whether
    it is open.

    Strategy (two passes, to cover both shooting styles the app allows):
      1. Run the eye cascade on the full frame directly. This is the
         common case for this app: a close-up macro photo of the eye/
         conjunctiva, with no full face in view.
      2. If that finds nothing, look for a face first and re-run the eye
         cascade restricted to the face region. This recovers cases
         where the user photographed their whole face instead of
         zooming into the eye, where a direct full-frame eye search is
         more prone to missing a comparatively small eye region.

    Runs on a single CPU-bound grayscale pass; no GPU or network access
    is required.
    """
    gray = _to_gray_array(image)

    eyes = _eye_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6, minSize=(40, 40)
    )
    if len(eyes) > 0:
        boxes = [tuple(int(v) for v in box) for box in eyes]
        eyes_open = any(
            _has_visible_pupil(gray[y : y + h, x : x + w]) for (x, y, w, h) in boxes
        )
        return EyeDetectionResult(
            eye_found=True,
            eyes_detected=len(boxes),
            face_detected=False,
            eyes_open=eyes_open,
            eye_boxes=boxes,
        )

    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    for (fx, fy, fw, fh) in faces:
        face_roi = gray[fy : fy + fh, fx : fx + fw]
        eyes_in_face = _eye_cascade.detectMultiScale(
            face_roi, scaleFactor=1.1, minNeighbors=6, minSize=(20, 20)
        )
        if len(eyes_in_face) > 0:
            boxes = [
                (int(fx + x), int(fy + y), int(w), int(h)) for (x, y, w, h) in eyes_in_face
            ]
            eyes_open = any(
                _has_visible_pupil(gray[y : y + h, x : x + w]) for (x, y, w, h) in boxes
            )
            return EyeDetectionResult(
                eye_found=True,
                eyes_detected=len(boxes),
                face_detected=True,
                eyes_open=eyes_open,
                eye_boxes=boxes,
            )

    return EyeDetectionResult(
        eye_found=False,
        eyes_detected=0,
        face_detected=len(faces) > 0,
        eyes_open=False,
        eye_boxes=[],
    )
