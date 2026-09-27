import cv2
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

while True:
    # Read frame from webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Run YOLO detection
    results = model(frame, conf=0.5, verbose=False)

    # Draw detections on frame
    annotated_frame = results[0].plot()

    # Display result
    cv2.imshow("YOLO Real-Time Object Detection", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

print("Detection stopped.")
