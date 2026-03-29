import cv2
import os
import numpy as np

dataset_path = "C:/Users/PAVAN/Documents/Projects/aiml_project/dataset"

faces = []
labels = []
label_map = {}

current_label = 0

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


print("Reading dataset...")

for file in os.listdir(dataset_path):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):

        path = os.path.join(dataset_path, file)
        print(f"\nProcessing: {file}")

        img = cv2.imread(path)

        if img is None:
            print("❌ Image not loaded")
            continue

        # Resize (important for detection)
        img = cv2.resize(img, (500, 500))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Extract name + USN
        try:
            name = file.split("_")[0]
            usn = file.split("_")[1]
        except:
            print("❌ Filename format wrong:", file)
            continue

        key = name + "_" + usn

        # Assign label
        if key not in label_map:
            label_map[key] = current_label
            current_label += 1

        label = label_map[key]

        # Detect faces
        faces_detected = face_cascade.detectMultiScale(gray, 1.2, 4)

        print("Faces found:", len(faces_detected))

        # Extract face
        for (x, y, w, h) in faces_detected:
            face = gray[y:y+h, x:x+w]
            faces.append(face)
            labels.append(label)


print("\nTotal faces collected:", len(faces))

if len(faces) == 0:
    print("❌ ERROR: No faces detected. Fix dataset.")
    exit()

print("\nTraining model...")

model = cv2.face.LBPHFaceRecognizer_create()
model.train(faces, np.array(labels))

model.save("face_model.yml")
np.save("labels.npy", label_map)

print("\n✅ Training completed successfully!")
print("Saved: face_model.yml + labels.npy")