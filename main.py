"""
Command-line interface for the CV Attendance System.
Use this if you prefer a terminal-based workflow instead of the web dashboard.

Usage:
    python main.py register    – Register a new student and capture face samples
    python main.py train       – Train the face recognition model
    python main.py recognize   – Start real-time attendance marking
    python main.py report      – Print today's attendance summary
    python main.py list        – List all registered students
    python main.py web         – Launch the web dashboard
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from student_db import StudentDB
from attendance_manager import AttendanceManager
from face_trainer import train_model, model_exists
from face_capture import capture_faces
from face_recognition_module import run_recognition


def cmd_register():
    print("\n=== Register New Student ===")
    name = input("Full Name       : ").strip()
    roll = input("Roll Number     : ").strip()
    dept = input("Department      : ").strip() or "Computer Science"

    if not name or not roll:
        print("[ERROR] Name and Roll Number are required.")
        return

    db = StudentDB()
    student_id = db.add(name, roll, dept)
    print(f"[OK] Registered '{name}' with ID={student_id}")
    print("[INFO] Opening webcam for face capture…")
    capture_faces(student_id, name)
    print("[INFO] Face capture complete. Run 'train' to update the model.")


def cmd_train():
    print("\n=== Training Face Recognition Model ===")
    success, count = train_model()
    if success:
        print(f"[OK] Model trained on {count} student(s).")
    else:
        print("[ERROR] Training failed. Register students with face samples first.")


def cmd_recognize():
    print("\n=== Starting Real-Time Attendance Recognition ===")
    if not model_exists():
        print("[ERROR] No trained model found. Run 'train' first.")
        return
    run_recognition()


def cmd_report():
    att = AttendanceManager()
    summary = att.get_summary()
    records = att.get_today()

    print(f"\n=== Attendance Report for {summary['date']} ===")
    print(f"  Total   : {summary['total']}")
    print(f"  Present : {summary['present']}")
    print(f"  Absent  : {summary['absent']}")
    print(f"  Rate    : {summary['percentage']}%")
    if not records.empty:
        print("\n  Present students:")
        for _, row in records.iterrows():
            print(f"    [{row['time']}] {row['name']} ({row['roll_number']})")


def cmd_list():
    db = StudentDB()
    students = db.get_all()
    if students.empty:
        print("No students registered.")
        return
    print(f"\n{'ID':<5} {'Name':<20} {'Roll No.':<15} {'Department'}")
    print("-" * 60)
    for _, s in students.iterrows():
        print(f"{int(s['id']):<5} {s['name']:<20} {s['roll_number']:<15} {s['department']}")
    print(f"\nTotal: {len(students)} student(s)")


def cmd_web():
    from app import app
    print("\n=== CV Attendance Web Dashboard ===")
    print("Open http://127.0.0.1:5000 in your browser\n")
    app.run(debug=False, host="0.0.0.0", port=5000)


COMMANDS = {
    "register": cmd_register,
    "train": cmd_train,
    "recognize": cmd_recognize,
    "report": cmd_report,
    "list": cmd_list,
    "web": cmd_web,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)
    COMMANDS[sys.argv[1]]()