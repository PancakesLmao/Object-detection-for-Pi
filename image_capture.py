import cv2
import os

# Change this to the name of the person you're photographing
PERSON_NAME = "Pascale Quester"

# Create output directory if it doesn't exist
output_dir = os.path.join("dataset", PERSON_NAME)
os.makedirs(output_dir, exist_ok=True)

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

print("[INFO] Press SPACE to capture, 'q' to quit...")

img_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow("Image Capture", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):
        img_path = os.path.join(output_dir, f"{img_count}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"[INFO] Saved {img_path}")
        img_count += 1
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
