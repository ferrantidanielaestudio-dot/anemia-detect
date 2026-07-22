"""
Pre-inference image validation checks.

This module holds lightweight, classical-CV gatekeeping checks that run
BEFORE the image reaches the anemia classification model. It is kept
separate from app.py so new checks (e.g. "is the conjunctiva visible /
not obstructed by an eyelid") can be added here later without touching
the FastAPI routing logic.
"""

import os
from dataclasses import dataclass

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


def _to_gray_array(image: Image.Image) -> np.ndarray:
    gray_image = image.convert("L")
    return np.array(gray_image)


def detect_eye(image: Image.Image) -> EyeDetectionResult:
    """
    Determines whether a human eye is present in the image.

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
        return EyeDetectionResult(eye_found=True, eyes_detected=len(eyes), face_detected=False)

    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    for (fx, fy, fw, fh) in faces:
        face_roi = gray[fy : fy + fh, fx : fx + fw]
        eyes_in_face = _eye_cascade.detectMultiScale(
            face_roi, scaleFactor=1.1, minNeighbors=6, minSize=(20, 20)
        )
        if len(eyes_in_face) > 0:
            return EyeDetectionResult(
                eye_found=True, eyes_detected=len(eyes_in_face), face_detected=True
            )

    return EyeDetectionResult(
        eye_found=False, eyes_detected=0, face_detected=len(faces) > 0
    )
