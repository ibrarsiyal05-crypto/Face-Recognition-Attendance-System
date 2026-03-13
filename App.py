"""
Flask web dashboard for the Attendance System.
Provides a browser UI to manage students and view attendance records.
"""
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
from datetime import date, datetime
import os
import threading

from student_db import StudentDB
from attendance_manager import AttendanceManager
from face_trainer import train_model, model_exists

# ---- optional imports (only used if webcam available) ----
try:
    from face_capture import capture_faces, delete_dataset
    from face_recognition_module import run_recognition
    WEBCAM_AVAILABLE = True
except Exception:
    WEBCAM_AVAILABLE = False

app = Flask(__name__)
db = StudentDB()
att = AttendanceManager()

# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    summary = att.get_summary()
    students = db.get_all().to_dict("records")
    recent = att.get_today().to_dict("records")[-10:][::-1]
    return render_template("index.html",
                           summary=summary,
                           students=students,
                           recent=recent,
                           model_ready=model_exists())


# ── Student management ──────────────────────────────────────

@app.route("/students")
def students():
    all_students = db.get_all().to_dict("records")
    return render_template("students.html", students=all_students)


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        roll = request.form.get("roll_number", "").strip()
        dept = request.form.get("department", "").strip()

        if not name or not roll:
            return render_template("register.html",
                                   error="Name and Roll Number are required.")

        student_id = db.add(name, roll, dept)

        if WEBCAM_AVAILABLE:
            # Run capture in background so Flask stays responsive
            t = threading.Thread(
                target=capture_faces, args=(student_id, name), daemon=True
            )
            t.start()
            msg = (f"Student '{name}' registered (ID={student_id}). "
                   "Face capture window should have opened.")
        else:
            msg = (f"Student '{name}' registered (ID={student_id}). "
                   "Webcam not available – add face images manually to dataset/.")

        return render_template("register.html", success=msg)

    return render_template("register.html")


@app.route("/delete_student/<int:student_id>", methods=["POST"])
def delete_student(student_id):
    db.delete(student_id)
    if WEBCAM_AVAILABLE:
        delete_dataset(student_id)
    return redirect(url_for("students"))


# ── Training ────────────────────────────────────────────────

@app.route("/train", methods=["POST"])
def train():
    success, count = train_model()
    if success:
        msg = f"Model trained successfully on {count} student(s)."
    else:
        msg = "Training failed – make sure students have registered face samples."
    return redirect(url_for("index") + f"?msg={msg}")


# ── Recognition ─────────────────────────────────────────────

@app.route("/start_recognition", methods=["POST"])
def start_recognition():
    if not WEBCAM_AVAILABLE:
        return redirect(url_for("index") + "?msg=Webcam+not+available.")
    if not model_exists():
        return redirect(url_for("index") + "?msg=Train+the+model+first.")
    t = threading.Thread(target=run_recognition, daemon=True)
    t.start()
    return redirect(url_for("index") + "?msg=Recognition+started+in+a+new+window.")


# ── Attendance records ───────────────────────────────────────

@app.route("/attendance")
def attendance():
    selected_date_str = request.args.get("date", date.today().strftime("%Y-%m-%d"))
    try:
        selected_date = datetime.strptime(selected_date_str, "%Y-%m-%d").date()
    except ValueError:
        selected_date = date.today()

    records = att.get_by_date(selected_date).to_dict("records")
    summary = att.get_summary(selected_date)
    all_dates = att.get_all_dates()
    return render_template("attendance.html",
                           records=records,
                           summary=summary,
                           all_dates=all_dates,
                           selected_date=selected_date_str)


@app.route("/export_csv")
def export_csv():
    date_str = request.args.get("date", date.today().strftime("%Y-%m-%d"))
    try:
        export_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        export_date = date.today()

    filepath = att.export_csv(export_date)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return "No attendance file found for that date.", 404


# ── API endpoints (JSON) ─────────────────────────────────────

@app.route("/api/today_summary")
def api_today_summary():
    return jsonify(att.get_summary())


@app.route("/api/mark/<int:student_id>", methods=["POST"])
def api_mark(student_id):
    result = att.mark(student_id)
    return jsonify(result)


if __name__ == "__main__":
    print("=" * 50)
    print("  CV Attendance System – Web Dashboard")
    print("  Open http://127.0.0.1:5000 in your browser")
    print("=" * 50)
    app.run(debug=True, host="0.0.0.0", port=5000)