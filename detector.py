
from collections import Counter

from ultralytics import YOLO


class ObjectDetector:
    """
    Handles YOLO model loading, object detection,
    statistics calculation, and visualization.
    """

    def __init__(self, model_path="yolo11n.pt"):
        """
        Load the YOLO model.
        """

        self.model = YOLO(model_path)

    def detect(self, image, confidence=0.5):
        """
        Run YOLO object detection on an image.

        Returns:
            YOLO result for the given image.
        """

        results = self.model(
            image,
            conf=confidence,
            verbose=False
        )

        return results[0]

    def get_statistics(self, result):
        """
        Calculate statistics from YOLO detections.

        Returns:
            Dictionary containing:
            - total_objects
            - object_counts
            - average_confidence
        """

        detected_objects = []
        confidence_values = []

        # Process every detected bounding box
        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )

            object_name = self.model.names[
                class_id
            ]

            detected_objects.append(
                object_name
            )

            confidence_values.append(
                confidence
            )

        # Count objects by class
        object_counts = Counter(
            detected_objects
        )

        # Total number of detections
        total_objects = len(
            detected_objects
        )

        # Calculate average confidence
        if confidence_values:

            average_confidence = (
                sum(confidence_values)
                / len(confidence_values)
            )

        else:

            average_confidence = 0.0

        return {
            "total_objects": total_objects,
            "object_counts": dict(object_counts),
            "average_confidence": average_confidence
        }

    def draw_detections(self, result):
        """
        Draw bounding boxes, class names,
        and confidence scores on the image.
        """

        return result.plot()
