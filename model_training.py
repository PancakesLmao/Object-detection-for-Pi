import os
import cv2
import face_recognition
import pickle

dataset_dir = "dataset"
encodings = []
names = []

print("[INFO] Starting training...")

for person_name in os.listdir(dataset_dir):
    person_path = os.path.join(dataset_dir, person_name)
    if not os.path.isdir(person_path):
        continue

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")
        encs = face_recognition.face_encodings(rgb, boxes)

        for encoding in encs:
            encodings.append(encoding)
            names.append(person_name)

print(f"[INFO] Trained on {len(encodings)} face(s).")

# Save encodings
with open("encodings.pickle", "wb") as f:
    pickle.dump({"encodings": encodings, "names": names}, f)

print("[INFO] Training complete. Model saved to encodings.pickle")
