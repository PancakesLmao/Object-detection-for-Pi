import face_recognition
import pickle
import os

# Path to your dataset
dataset_path = "/home/pancakes/Desktop/Object-detection-for-Pi/dataset"  # Replace with your dataset path

# Lists to store encodings and names
known_encodings = []
known_names = []

# Iterate through the dataset
print("[INFO] Processing dataset...")
for person_name in os.listdir(dataset_path):
    person_dir = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_dir):
        continue
    for image_name in os.listdir(person_dir):
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        image_path = os.path.join(person_dir, image_name)
        print(f"Processing {image_path}...")
        # Load the image
        image = face_recognition.load_image_file(image_path)
        # Get face encodings
        encodings = face_recognition.face_encodings(image)
        if len(encodings) > 0:
            known_encodings.append(encodings[0])  # Assume one face per image
            known_names.append(person_name)
        else:
            print(f"[WARNING] No faces found in {image_path}")

# Save encodings and names to pickle file
data = {"encodings": known_encodings, "names": known_names}
with open("encodings.pickle", "wb") as f:
    pickle.dump(data, f)
print("[INFO] Encodings saved to encodings.pickle")
