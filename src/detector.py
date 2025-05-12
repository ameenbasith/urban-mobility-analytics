import torch
from pathlib import Path
import cv2
import numpy as np
import time
from ultralytics import YOLO


class ObjectDetector:
    def __init__(self, model_path=None):
        """Initialize the YOLO object detector.

        Args:
            model_path: Path to custom model weights or model name
        """
        # Load a pretrained YOLOv5 model
        if model_path:
            self.model = YOLO(model_path)
        else:
            # Use the default YOLOv5s model
            self.model = YOLO('yolov8n.pt')

        # Classes we're interested in (from COCO dataset)
        # 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
        self.target_classes = [0, 1, 2, 3, 5, 7]

        # Class name mapping
        self.class_names = {
            0: 'person',
            1: 'bicycle',
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }

        print("Object detector initialized")

    def detect(self, image, confidence_threshold=0.3):
        """Run object detection on an image.

        Args:
            image: OpenCV image in BGR format
            confidence_threshold: Minimum confidence score

        Returns:
            Dictionary with detection results
        """
        # YOLOv8 expects RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get predictions
        start_time = time.time()
        results = self.model.predict(img_rgb, conf=confidence_threshold)
        inference_time = time.time() - start_time

        # Process results
        detections = []

        # Get detection boxes
        result = results[0]  # First image result

        # Extract boxes, classes and confidence scores
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = box.conf[0].item()

            # Only keep our target classes
            if class_id in self.target_classes:
                # Get box coordinates [x1, y1, x2, y2]
                coords = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, coords)

                detections.append({
                    'class_id': class_id,
                    'class_name': self.class_names.get(class_id, 'unknown'),
                    'confidence': confidence,
                    'box': [x1, y1, x2, y2],
                    'width': x2 - x1,
                    'height': y2 - y1,
                    'center': [(x1 + x2) // 2, (y1 + y2) // 2]
                })

        return {
            'detections': detections,
            'inference_time': inference_time,
            'image_size': image.shape[:2]  # (height, width)
        }

    def draw_detections(self, image, detections):
        """Draw detection boxes and labels on image.

        Args:
            image: Original image
            detections: Detection results from detect()

        Returns:
            Image with drawn detections
        """
        img_copy = image.copy()

        # Color mapping for different classes (BGR format)
        colors = {
            0: (0, 255, 0),  # person: green
            1: (255, 0, 0),  # bicycle: blue
            2: (0, 0, 255),  # car: red
            3: (255, 255, 0),  # motorcycle: cyan
            5: (255, 0, 255),  # bus: magenta
            7: (0, 255, 255)  # truck: yellow
        }

        # Draw each detection
        for det in detections['detections']:
            # Extract info
            box = det['box']
            label = f"{det['class_name']} {det['confidence']:.2f}"
            color = colors.get(det['class_id'], (0, 255, 0))

            # Draw bounding box
            cv2.rectangle(img_copy, (box[0], box[1]), (box[2], box[3]), color, 2)

            # Draw label background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(img_copy, (box[0], box[1] - text_size[1] - 10),
                          (box[0] + text_size[0], box[1]), color, -1)

            # Draw label text
            cv2.putText(img_copy, label, (box[0], box[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Draw center point
            center = det['center']
            cv2.circle(img_copy, (center[0], center[1]), 3, color, -1)

        # Add inference time
        cv2.putText(img_copy, f"Inference: {detections['inference_time']:.3f}s",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return img_copy


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = ObjectDetector()

    # Load test image
    image_path = "../data/frames/frame_00000.jpg"
    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not load image: {image_path}")
    else:
        # Run detection
        results = detector.detect(image)

        # Draw results
        output_image = detector.draw_detections(image, results)

        # Save the result
        output_path = "../results/detection_example.jpg"
        Path("../results").mkdir(exist_ok=True)
        cv2.imwrite(output_path, output_image)

        print(f"Detection complete. Found {len(results['detections'])} objects.")
        print(f"Result saved to {output_path}")