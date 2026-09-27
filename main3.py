import cv2
import time
from collections import Counter
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

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    # -------------------------
    # OBJECT COUNTING
    # -------------------------

    detected_objects = []

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        object_name = model.names[class_id]

        detected_objects.append(object_name)

    # Count each object
    object_counts = Counter(detected_objects)

    # Total objects
    total_objects = len(detected_objects)

    # -------------------------
    # FPS CALCULATION
    # -------------------------

    current_time = time.time()

    if prev_time != 0:
        fps = 1 / (current_time - prev_time)
    else:
        fps = 0

    prev_time = current_time

    # -------------------------
    # DISPLAY FPS
    # -------------------------

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # -------------------------
    # DISPLAY OBJECT COUNT
    # -------------------------

    cv2.putText(
        annotated_frame,
        f"Objects: {total_objects}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # -------------------------
    # DISPLAY INDIVIDUAL COUNTS
    # -------------------------

    y_position = 120

    for object_name, count in object_counts.items():

        text = f"{object_name}: {count}"

        cv2.putText(
            annotated_frame,
            text,
            (20, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        y_position += 30

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
