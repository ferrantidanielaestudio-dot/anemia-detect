"""
Pre-inference image validation checks.

This module holds lightweight, classical-CV gatekeeping checks that run
BEFORE the image reaches the anemia classification model. It is kept
separate from app.py so new checks can be added here later without
touching the FastAPI routing logic.

Checks implemented so far:
  - detect_eye(): is there a human eye in the frame at all?
  - eyes_open: is the eye actually open (pupil visible)?
  - conjunctiva_visible: is there enough exposed space *below* the pupil,
    and is the shot a close-up (not a distant face photo)? The model was
    trained on the lower palpebral conjunctiva specifically — a photo of
    just the upper eye/iris, or a wide face shot where the eye is tiny,
    doesn't show the tissue the model actually needs.
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

# Tuning knobs for the geometric conjunctiva-framing checks. Kept as
# module-level constants (not magic numbers buried in the function) so
# they're easy to retune later against real photos.
MIN_EYE_WIDTH_RATIO = 0.15    # eye bbox width / image width -> "is this a close-up?"
MIN_SPACE_BELOW_PUPIL_RATIO = 0.28  # (eye bbox bottom - pupil bottom) / eye bbox height


@dataclass
class EyeDetectionResult:
    eye_found: bool
    eyes_detected: int
    face_detected: bool
    eyes_open: bool
    conjunctiva_visible: bool
    eye_boxes: list = field(default_factory=list)  # [(x, y, w, h), ...] in `gray` coordinates


def _to_gray_array(image: Image.Image) -> np.ndarray:
    gray_image = image.convert("L")
    return np.array(gray_image)


def _find_pupil_circle(eye_roi: np.ndarray):
    """
    Locates the pupil/iris inside a detected eye region: a small, dark,
    roughly circular blob against the lighter sclera. A closed eyelid has
    no such structure (just skin/eyelash texture), so this reliably tells
    open from closed without any color rules — it looks at shape (via
    Hough circle transform) and local contrast, not hue.

    Returns (cx, cy, r) in eye_roi-local coordinates, or None if no pupil
    is found.
    """
    h, w = eye_roi.shape[:2]
    if h < 10 or w < 10:
        return None

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
        return None

    # A real pupil is noticeably darker than the eye region around it.
    # This rejects circles Hough finds on eyelid folds/eyelashes in a
    # closed eye, where there is no strong dark blob at all.
    roi_mean = float(np.mean(eye_roi))
    for cx, cy, r in circles[0]:
        cx_i, cy_i, half = int(cx), int(cy), max(1, int(r * 0.6))
        y0, y1 = max(0, cy_i - half), min(h, cy_i + half)
        x0, x1 = max(0, cx_i - half), min(w, cx_i + half)
        patch = eye_roi[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        if float(np.mean(patch)) < roi_mean - 15:
            return (cx_i, cy_i, int(r))

    return None


def _assess_eye_region(
    gray: np.ndarray, box: tuple, image_width: int
) -> tuple:
    """
    Given one detected eye bounding box, returns (eyes_open, conjunctiva_visible)
    for that region.
    """
    x, y, w, h = box
    eye_roi = gray[y : y + h, x : x + w]

    pupil = _find_pupil_circle(eye_roi)
    if pupil is None:
        return False, False

    _cx, cy, r = pupil
    pupil_bottom = cy + r

    is_close_up = (w / image_width) >= MIN_EYE_WIDTH_RATIO
    space_below_ratio = (h - pupil_bottom) / h if h > 0 else 0.0
    has_room_below = space_below_ratio >= MIN_SPACE_BELOW_PUPIL_RATIO

    return True, (is_close_up and has_room_below)


def detect_eye(image: Image.Image) -> EyeDetectionResult:
    """
    Determines whether a human eye is present in the image, whether it is
    open, and whether the lower conjunctiva is plausibly in frame (close-up
    shot, with enough space below the pupil for the exposed tissue the
    model was trained on).

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
    image_width = gray.shape[1]

    eyes = _eye_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6, minSize=(40, 40)
    )
    if len(eyes) > 0:
        boxes = [tuple(int(v) for v in box) for box in eyes]
        assessments = [_assess_eye_region(gray, box, image_width) for box in boxes]
        eyes_open = any(open_ for open_, _ in assessments)
        conjunctiva_visible = any(visible for _, visible in assessments)
        return EyeDetectionResult(
            eye_found=True,
            eyes_detected=len(boxes),
            face_detected=False,
            eyes_open=eyes_open,
            conjunctiva_visible=conjunctiva_visible,
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
            assessments = [_assess_eye_region(gray, box, image_width) for box in boxes]
            eyes_open = any(open_ for open_, _ in assessments)
            conjunctiva_visible = any(visible for _, visible in assessments)
            return EyeDetectionResult(
                eye_found=True,
                eyes_detected=len(boxes),
                face_detected=True,
                eyes_open=eyes_open,
                conjunctiva_visible=conjunctiva_visible,
                eye_boxes=boxes,
            )

    return EyeDetectionResult(
        eye_found=False,
        eyes_detected=0,
        face_detected=len(faces) > 0,
        eyes_open=False,
        conjunctiva_visible=False,
        eye_boxes=[],
    )
