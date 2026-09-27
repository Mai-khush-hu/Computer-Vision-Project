import cv2
from ultralytics import YOLO
from collections import Counter


# --------------------------------
# LOAD YOLO MODEL
# --------------------------------

model = YOLO("yolo11n.pt")


# --------------------------------
# WEBCAM DETECTION
# --------------------------------

def webcam_detection():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("\nWebcam detection started.")
    print("Press Q to quit.")

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read webcam frame.")
            break

        results = model(
            frame,
            conf=0.5,
            verbose=False
        )

        annotated_frame = results[0].plot()

        cv2.imshow(
            "Webcam Object Detection",
            annotated_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# --------------------------------
# IMAGE DETECTION
# --------------------------------

def image_detection():

    image_path = input(
        "\nEnter image path: "
    )

    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not load image.")
        return

    results = model(
        image,
        conf=0.5,
        verbose=False
    )

    annotated_image = results[0].plot()

    # Count objects
    detected_objects = []

    for box in results[0].boxes:

        class_id = int(box.cls[0])

        object_name = model.names[class_id]

        detected_objects.append(object_name)

    object_counts = Counter(detected_objects)

    print("\nDetected Objects:")

    for object_name, count in object_counts.items():

        print(
            f"{object_name}: {count}"
        )

    # Display image
    cv2.imshow(
        "Image Object Detection",
        annotated_image
    )

    print("\nPress any key to close.")

    cv2.waitKey(0)

    cv2.destroyAllWindows()


# --------------------------------
# VIDEO DETECTION
# --------------------------------

def video_detection():

    video_path = input(
        "\nEnter video path: "
    )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    print("\nVideo detection started.")
    print("Press Q to quit.")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = model(
            frame,
            conf=0.5,
            verbose=False
        )

        annotated_frame = results[0].plot()

        cv2.imshow(
            "Video Object Detection",
            annotated_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# --------------------------------
# MAIN MENU
# --------------------------------

while True:

    print("\n==============================")
    print(" AI OBJECT DETECTION SYSTEM")
    print("==============================")
    print("1. Webcam")
    print("2. Image")
    print("3. Video")
    print("4. Exit")

    choice = input(
        "\nEnter your choice: "
    )

    if choice == "1":

        webcam_detection()

    elif choice == "2":

        image_detection()

    elif choice == "3":

        video_detection()

    elif choice == "4":

        print("\nProgram exited.")
        break

    else:

        print("\nInvalid choice. Please try again.")
