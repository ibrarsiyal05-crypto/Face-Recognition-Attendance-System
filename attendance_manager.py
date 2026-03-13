"""
Attendance records manager.
Each day's attendance is stored in a separate CSV file.
"""
import os
import pandas as pd
from datetime import datetime, date
from config import ATTENDANCE_DIR
from student_db import StudentDB


class AttendanceManager:
    COLUMNS = ["student_id", "name", "roll_number", "department",
               "date", "time", "status"]

    def __init__(self):
        self.db = StudentDB()

    def _get_file(self, for_date: date = None) -> str:
        """Return the attendance CSV path for a given date."""
        if for_date is None:
            for_date = date.today()
        filename = f"attendance_{for_date.strftime('%Y_%m_%d')}.csv"
        return os.path.join(ATTENDANCE_DIR, filename)

    def _load(self, for_date: date = None) -> pd.DataFrame:
        filepath = self._get_file(for_date)
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        return pd.DataFrame(columns=self.COLUMNS)

    def _save(self, df: pd.DataFrame, for_date: date = None):
        filepath = self._get_file(for_date)
        df.to_csv(filepath, index=False)

    def mark(self, student_id: int) -> dict:
        """
        Mark attendance for a student.
        Returns dict with 'success', 'already_marked', and 'student' keys.
        """
        student = self.db.get_by_id(student_id)
        if not student:
            return {"success": False, "already_marked": False, "student": None}

        today = date.today()
        df = self._load(today)

        already = not df[df["student_id"] == student_id].empty

        if not already:
            now = datetime.now()
            new_row = {
                "student_id": student_id,
                "name": student["name"],
                "roll_number": student["roll_number"],
                "department": student["department"],
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "status": "Present",
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            self._save(df, today)

        return {
            "success": True,
            "already_marked": already,
            "student": student,
        }

    def get_today(self) -> pd.DataFrame:
        return self._load(date.today())

    def get_by_date(self, for_date: date) -> pd.DataFrame:
        return self._load(for_date)

    def get_all_dates(self) -> list[str]:
        """Return all dates that have attendance records."""
        dates = []
        for f in os.listdir(ATTENDANCE_DIR):
            if f.startswith("attendance_") and f.endswith(".csv"):
                date_str = f.replace("attendance_", "").replace(".csv", "")
                dates.append(date_str.replace("_", "-"))
        return sorted(dates, reverse=True)

    def get_summary(self, for_date: date = None) -> dict:
        df = self._load(for_date)
        total_students = self.db.count()
        present = len(df)
        absent = max(0, total_students - present)
        return {
            "date": (for_date or date.today()).strftime("%Y-%m-%d"),
            "total": total_students,
            "present": present,
            "absent": absent,
            "percentage": round((present / total_students * 100) if total_students > 0 else 0, 1),
        }

    def export_csv(self, for_date: date = None) -> str:
        """Return the path of the attendance CSV for a given date."""
        return self._get_file(for_date)