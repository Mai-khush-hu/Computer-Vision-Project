import cv2
import time
from collections import Counter
from ultralytics import YOLO

# -----------------------------
# LOAD YOLO MODEL
# -----------------------------

model = YOLO("yolo11n.pt")

# -----------------------------
# START WEBCAM
# -----------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("AI Object Detection Started!")
print("Press Q to quit.")

# -----------------------------
# FPS VARIABLES
# -----------------------------

prev_time = 0

while True:

    # Read webcam frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # -----------------------------
    # YOLO DETECTION
    # -----------------------------

    results = model(
        frame,
        conf=0.5,
        verbose=False
    )

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    # -----------------------------
    # OBJECT INFORMATION
    # -----------------------------

    detected_objects = []
    confidence_values = []

    for box in results[0].boxes:

        class_id = int(box.cls[0])

        confidence = float(box.conf[0])

        object_name = model.names[class_id]

        detected_objects.append(object_name)

        confidence_values.append(confidence)

    # Count objects
    object_counts = Counter(detected_objects)

    # Total number of objects
    total_objects = len(detected_objects)

    # -----------------------------
    # AVERAGE CONFIDENCE
    # -----------------------------

    if confidence_values:

        average_confidence = (
            sum(confidence_values)
            / len(confidence_values)
        )

    else:

        average_confidence = 0

    # -----------------------------
    # FPS
    # -----------------------------

    current_time = time.time()

    if prev_time != 0:
        fps = 1 / (current_time - prev_time)
    else:
        fps = 0

    prev_time = current_time

    # -----------------------------
    # INFORMATION PANEL
    # -----------------------------

    panel_width = 260
    panel_height = 180

    overlay = annotated_frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (panel_width, panel_height),
        (0, 0, 0),
        -1
    )

    # Make panel slightly transparent
    cv2.addWeighted(
        overlay,
        0.65,
        annotated_frame,
        0.35,
        0,
        annotated_frame
    )

    # -----------------------------
    # PANEL TEXT
    # -----------------------------

    cv2.putText(
        annotated_frame,
        "AI OBJECT DETECTION",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (15, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Objects: {total_objects}",
        (15, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Avg Confidence: {average_confidence * 100:.1f}%",
        (15, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    # -----------------------------
    # OBJECT COUNTS
    # -----------------------------

    y_position = 155

    for object_name, count in object_counts.items():

        text = f"{object_name}: {count}"

        cv2.putText(
            annotated_frame,
            text,
            (15, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        y_position += 25

    # -----------------------------
    # DISPLAY
    # -----------------------------

    cv2.imshow(
        "AI Object Detection System",
        annotated_frame
    )

    # -----------------------------
    # QUIT
    # -----------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -----------------------------
# RELEASE RESOURCES
# -----------------------------

cap.release()
cv2.destroyAllWindows()

print("Detection stopped.")
