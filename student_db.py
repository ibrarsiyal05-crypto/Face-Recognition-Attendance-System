"""
Student database manager using CSV storage.
Handles CRUD operations for student records.
"""
import csv
import os
import pandas as pd
from datetime import datetime
from config import STUDENTS_FILE


class StudentDB:
    COLUMNS = ["id", "name", "roll_number", "department", "registered_on"]

    def __init__(self):
        self._ensure_file()

    def _ensure_file(self):
        """Create students CSV if it doesn't exist."""
        if not os.path.exists(STUDENTS_FILE):
            df = pd.DataFrame(columns=self.COLUMNS)
            df.to_csv(STUDENTS_FILE, index=False)

    def get_all(self) -> pd.DataFrame:
        """Return all students as a DataFrame."""
        try:
            return pd.read_csv(STUDENTS_FILE)
        except Exception:
            return pd.DataFrame(columns=self.COLUMNS)

    def get_by_id(self, student_id: int) -> dict | None:
        """Return a student record by face-ID."""
        df = self.get_all()
        row = df[df["id"] == student_id]
        if row.empty:
            return None
        return row.iloc[0].to_dict()

    def get_next_id(self) -> int:
        """Return the next available integer ID."""
        df = self.get_all()
        if df.empty:
            return 1
        return int(df["id"].max()) + 1

    def add(self, name: str, roll_number: str, department: str) -> int:
        """Add a new student and return their assigned ID."""
        df = self.get_all()
        # Check for duplicate roll number
        if not df.empty and roll_number in df["roll_number"].values:
            existing = df[df["roll_number"] == roll_number].iloc[0]
            return int(existing["id"])  # return existing ID

        new_id = self.get_next_id()
        new_row = {
            "id": new_id,
            "name": name,
            "roll_number": roll_number,
            "department": department,
            "registered_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(STUDENTS_FILE, index=False)
        return new_id

    def delete(self, student_id: int) -> bool:
        """Remove a student by ID."""
        df = self.get_all()
        initial_len = len(df)
        df = df[df["id"] != student_id]
        if len(df) < initial_len:
            df.to_csv(STUDENTS_FILE, index=False)
            return True
        return False

    def get_name(self, student_id: int) -> str:
        """Return just the name for a given ID, or 'Unknown'."""
        student = self.get_by_id(student_id)
        return student["name"] if student else "Unknown"

    def count(self) -> int:
        return len(self.get_all())