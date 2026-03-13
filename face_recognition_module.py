"""
Face recognition module — uses EigenFace (PCA + cosine similarity) recognizer.
Press 'q' in the webcam window to quit.
"""
import cv2
import time
from config import (
    RECOGNITION_COOLDOWN, FRAME_WIDTH, FRAME_HEIGHT, CAMERA_INDEX,
    COLOR_GREEN, COLOR_RED, COLOR_WHITE, COLOR_YELLOW, COLOR_ORANGE,
)
from student_db import StudentDB
from attendance_manager import AttendanceManager
from face_trainer import model_exists
from face_recognizer import EigenFaceRecognizer, model_file_path

COSINE_THRESHOLD = 0.35   # lower = stricter; tune per environment


def run_recognition():
    if not model_exists():
        print("[ERROR] No trained model. Run train first.")
        return

    recognizer = EigenFaceRecognizer()
    if not recognizer.load(model_file_path()):
        print("[ERROR] Failed to load model.")
        return

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    db = StudentDB()
    attendance = AttendanceManager()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    last_marked: dict = {}
    notification = None
    print("[INFO] Recognition started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            student_id, distance = recognizer.predict(face_roi)
            is_recognized = distance < COSINE_THRESHOLD

            if is_recognized:
                name = db.get_name(student_id)
                now = time.time()
                if student_id not in last_marked or (now - last_marked[student_id]) > RECOGNITION_COOLDOWN:
                    result = attendance.mark(student_id)
                    last_marked[student_id] = now
                    if result["success"]:
                        color = COLOR_GREEN if not result["already_marked"] else COLOR_ORANGE
                        msg = f"{name} – {'Already marked' if result['already_marked'] else 'Attendance Marked!'}"
                        notification = (msg, color, now + 3)

                label = f"{name} ({int((1-distance)*100)}%)"
                box_color = COLOR_GREEN
            else:
                label = "Unknown"
                box_color = COLOR_RED

            cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
            cv2.rectangle(frame, (x, y-28), (x+w, y), box_color, -1)
            cv2.putText(frame, label, (x+4, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)

        # HUD
        summary = attendance.get_summary()
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (265, 80), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        for i, line in enumerate([
            f"Date: {summary['date']}",
            f"Present: {summary['present']}  Absent: {summary['absent']}",
            f"Total: {summary['total']}  Rate: {summary['percentage']}%",
        ]):
            cv2.putText(frame, line, (8, 20+i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_YELLOW, 1)

        # Notification
        if notification:
            msg, color, expiry = notification
            if time.time() < expiry:
                cv2.rectangle(frame, (0, FRAME_HEIGHT-50), (FRAME_WIDTH, FRAME_HEIGHT), color, -1)
                cv2.putText(frame, msg, (10, FRAME_HEIGHT-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_WHITE, 2)
            else:
                notification = None

        cv2.imshow("Attendance System – Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Recognition stopped.")