import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
import time

# -------------------------------
# LOAD DATA
# -------------------------------
timetable = pd.read_csv("timetable.csv")
students = pd.read_csv("students.csv")

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def get_student_batch(usn):
    row = students[students['USN'] == usn]
    if not row.empty:
        return row.iloc[0]['Batch']
    return None

def get_current_subject(day, current_time, batch):
    for _, row in timetable.iterrows():
        if row['Day'] == day:

            if row['Batch'] != "ALL" and row['Batch'] != batch:
                continue

            if row['Start_Time'] <= current_time <= row['End_Time']:
                return row['Subject'], row['Start_Time']

    return None, None

# -------------------------------
# LOAD MODEL
# -------------------------------
model = cv2.face.LBPHFaceRecognizer_create()
model.read("face_model.yml")

label_map = np.load("labels.npy", allow_pickle=True).item()
label_map = {v: k for k, v in label_map.items()}

# -------------------------------
# ATTENDANCE FILE
# -------------------------------
ATTENDANCE_FILE = "attendance.csv"

if not os.path.exists(ATTENDANCE_FILE) or os.stat(ATTENDANCE_FILE).st_size == 0:
    df = pd.DataFrame(columns=[
    "Name", "USN", "Date",
    "Entry_Time", "Exit_Time",
    "Subject", "Late",
    "Duration_Minutes", "Status"
    ])
    df.to_csv(ATTENDANCE_FILE, index=False)
else:
    df = pd.read_csv(ATTENDANCE_FILE)

# -------------------------------
# TRACK LAST SEEN (FOR 1 MIN RULE)
# -------------------------------
last_seen = {}

# -------------------------------
# FACE DETECTOR
# -------------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# -------------------------------
# CAMERA
# -------------------------------
cap = cv2.VideoCapture(0)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]

        label, confidence = model.predict(face)

        if confidence < 70:
            student = label_map[label]
            name, usn = student.split("_")

            now = datetime.now()
            date = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M")
            day = now.strftime("%A")

            batch = get_student_batch(usn)

            subject, start_time = get_current_subject(day, time_str, batch)

            if subject is None:
                subject = "No Class"

            # -------------------------------
            # LATE LOGIC
            # -------------------------------
            late_status = "No"

            if start_time:
                start_dt = datetime.strptime(start_time, "%H:%M")
                current_dt = datetime.strptime(time_str, "%H:%M")

                diff = (current_dt - start_dt).seconds / 60

                if diff > 5:
                    late_status = "Yes"

            # -------------------------------
            # ENTRY / EXIT LOGIC (1 MIN RULE)
            # -------------------------------
            current_timestamp = time.time()

            existing = df[(df['USN'] == usn) & (df['Date'] == date)]

            if existing.empty:
                # 🟢 ENTRY
                new_row = {
                    "Name": name,
                    "USN": usn,
                    "Date": date,
                    "Entry_Time": time_str,
                    "Exit_Time": "",
                    "Subject": subject,
                    "Late": late_status,
                    "Duration_Minutes": "",
                    "Status": ""
                }

                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                last_seen[usn] = current_timestamp

                print(f"{name} ENTRY at {time_str}")

            else:
                # 🔴 EXIT (only after 1 minute)
                if usn not in last_seen:
                    last_seen[usn] = current_timestamp

                if current_timestamp - last_seen[usn] > 10:
                    index = existing.index[-1]
                    df.at[index, 'Exit_Time'] = time_str

                    # 🔥 CALCULATE DURATION
                    entry_time = df.at[index, 'Entry_Time']

                    t1 = datetime.strptime(entry_time, "%H:%M")
                    t2 = datetime.strptime(time_str, "%H:%M")

                    duration = int((t2 - t1).seconds / 60)
                    df.at[index, 'Duration_Minutes'] = duration

                    # 🔥 STATUS LOGIC
                    if duration >= 40:
                        status = "Present"
                    else:
                        status = "Partial"

                    df.at[index, 'Status'] = status

                    last_seen[usn] = current_timestamp

                    print(f"{name} EXIT at {time_str} | Duration: {duration} mins | {status}")

            df.to_csv(ATTENDANCE_FILE, index=False)

            display_text = f"{name} ({confidence:.0f})"

        else:
            display_text = "Unknown"

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, display_text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()