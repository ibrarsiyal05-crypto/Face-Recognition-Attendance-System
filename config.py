"""
Configuration settings for the CV Attendance System
"""
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAINER_DIR = os.path.join(BASE_DIR, "trainer")
ATTENDANCE_DIR = os.path.join(BASE_DIR, "attendance_records")
TRAINER_FILE = os.path.join(TRAINER_DIR, "trainer.yml")
STUDENTS_FILE = os.path.join(BASE_DIR, "students.csv")

# OpenCV
CASCADE_PATH = os.path.join(os.path.dirname(__file__),
                             "haarcascade_frontalface_default.xml")

# Face recognition settings
CONFIDENCE_THRESHOLD = 70    # lower = stricter (LBPH: lower distance = better match)
SAMPLE_COUNT = 30            # number of face samples to capture per student
RECOGNITION_COOLDOWN = 5     # seconds between marking same person again

# Camera
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Colors (BGR)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_ORANGE = (0, 165, 255)

# Ensure directories exist
for d in [DATASET_DIR, TRAINER_DIR, ATTENDANCE_DIR]:
    os.makedirs(d, exist_ok=True)