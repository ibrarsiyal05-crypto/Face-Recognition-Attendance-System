"""
Face capture module.
Captures face samples from a webcam and saves them to the dataset directory.
"""
import cv2
import os
from config import (
    DATASET_DIR, CASCADE_PATH, SAMPLE_COUNT,
    FRAME_WIDTH, FRAME_HEIGHT,
    COLOR_GREEN, COLOR_RED, COLOR_WHITE, COLOR_YELLOW, COLOR_ORANGE,
    CAMERA_INDEX,
)


def _get_cascade():
    """Load the Haar Cascade; fall back to cv2.data path."""
    if os.path.exists(CASCADE_PATH):
        path = CASCADE_PATH
    else:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(path)
    if detector.empty():
        raise RuntimeError(f"Failed to load cascade from: {path}")
    return detector


def capture_faces(student_id: int, student_name: str) -> bool:
    """
    Open webcam, detect face, and save SAMPLE_COUNT grayscale images.

    Returns True on success, False if camera could not be opened.
    """
    detector = _get_cascade()
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    count = 0
    print(f"[INFO] Starting face capture for: {student_name} (ID={student_id})")
    print("[INFO] Press 'q' to abort early.")

    while count < SAMPLE_COUNT:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(80, 80),
        )

        for (x, y, w, h) in faces:
            count += 1
            # Save face ROI
            face_img = gray[y:y + h, x:x + w]
            filename = os.path.join(
                DATASET_DIR, f"User.{student_id}.{count}.jpg"
            )
            cv2.imwrite(filename, face_img)

            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_GREEN, 2)

            # Progress bar
            progress = int((count / SAMPLE_COUNT) * (FRAME_WIDTH - 40))
            cv2.rectangle(frame, (20, FRAME_HEIGHT - 40),
                          (20 + progress, FRAME_HEIGHT - 20), COLOR_GREEN, -1)
            cv2.rectangle(frame, (20, FRAME_HEIGHT - 40),
                          (FRAME_WIDTH - 20, FRAME_HEIGHT - 20), COLOR_WHITE, 1)

            # Labels
            cv2.putText(frame, f"Capturing: {student_name}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_YELLOW, 2)
            cv2.putText(frame, f"Samples: {count}/{SAMPLE_COUNT}",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)

        if len(faces) == 0:
            cv2.putText(frame, "No face detected – adjust position",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)

        cv2.imshow("Face Capture – Press Q to abort", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Captured {count} samples for {student_name}.")
    return count > 0


def delete_dataset(student_id: int):
    """Remove all face images for a student from the dataset."""
    removed = 0
    for f in os.listdir(DATASET_DIR):
        if f.startswith(f"User.{student_id}."):
            os.remove(os.path.join(DATASET_DIR, f))
            removed += 1
    print(f"[INFO] Removed {removed} images for ID={student_id}.")