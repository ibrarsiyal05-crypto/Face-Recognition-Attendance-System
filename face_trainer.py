"""
Face trainer module.
Trains the EigenFace recognizer on all saved face samples in the dataset directory.
"""
import cv2
import os
from config import DATASET_DIR
from face_recognizer import EigenFaceRecognizer, model_file_path


def train_model() -> tuple[bool, int]:
    """
    Train the face recognizer on the dataset directory.
    Returns (success: bool, num_unique_students: int).
    """
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces, ids = _get_images_and_labels(DATASET_DIR, detector)

    if len(faces) == 0:
        print("[WARNING] No face samples found. Register students first.")
        return False, 0

    recognizer = EigenFaceRecognizer()
    recognizer.train(faces, ids)
    recognizer.save(model_file_path())

    unique_ids = len(set(ids))
    print(f"[INFO] Model trained on {len(faces)} images "
          f"from {unique_ids} student(s). Saved to {model_file_path()}")
    return True, unique_ids


def _get_images_and_labels(path: str, detector) -> tuple[list, list]:
    """Walk the dataset directory, detect faces, return (images, labels)."""
    image_paths = [
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    face_samples: list = []
    ids: list = []

    for image_path in image_paths:
        filename = os.path.basename(image_path)
        parts = filename.split(".")
        if len(parts) < 3:
            continue
        try:
            student_id = int(parts[1])
        except ValueError:
            continue

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        faces = detector.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            face_samples.append(img[y:y + h, x:x + w])
            ids.append(student_id)

        # Pre-cropped ROI saved directly – use as-is
        if len(faces) == 0:
            face_samples.append(img)
            ids.append(student_id)

    return face_samples, ids


def model_exists() -> bool:
    return os.path.exists(model_file_path())