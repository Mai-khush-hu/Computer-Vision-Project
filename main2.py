import cv2
import time
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Start webcam
cap = cv2.VideoCapture(0)

# Check webcam
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("YOLO Object Detection Started!")
print("Press Q to quit.")

# FPS variables
prev_time = 0

while True:

    # Read frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Run YOLO detection
    results = model(frame, conf=0.5, verbose=False)

    # Draw detections
    annotated_frame = results[0].plot()

    # Calculate FPS
    current_time = time.time()

    if prev_time != 0:
        fps = 1 / (current_time - prev_time)
    else:
        fps = 0

    prev_time = current_time

    # Display FPS
    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Show frame
    cv2.imshow(
        "YOLO Real-Time Object Detection",
        annotated_frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

print("Detection stopped.")
