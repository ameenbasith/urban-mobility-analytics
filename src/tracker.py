import cv2
import numpy as np
from collections import defaultdict


class ObjectTracker:
    def __init__(self, max_disappeared=10, min_distance=50):
        """Initialize the object tracker.

        Args:
            max_disappeared: Maximum number of frames an object can disappear
            min_distance: Minimum distance for considering a match
        """
        self.next_object_id = 0
        self.objects = {}  # Dictionary of tracked objects
        self.disappeared = defaultdict(int)
        self.tracks = defaultdict(list)  # Store object tracks

        self.max_disappeared = max_disappeared
        self.min_distance = min_distance

        print("Object tracker initialized")

    def register(self, centroid, class_id, bbox, confidence):
        """Register a new object.

        Args:
            centroid: (x, y) center point of the object
            class_id: Class ID of the object
            bbox: Bounding box as [x1, y1, x2, y2]
            confidence: Detection confidence
        """
        object_id = self.next_object_id
        self.objects[object_id] = {
            'centroid': centroid,
            'class_id': class_id,
            'bbox': bbox,
            'confidence': confidence
        }
        self.disappeared[object_id] = 0
        self.tracks[object_id].append(centroid)
        self.next_object_id += 1

        return object_id

    def deregister(self, object_id):
        """Deregister an object.

        Args:
            object_id: ID of the object to deregister
        """
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections):
        """Update tracked objects with new detections.

        Args:
            detections: List of detection dictionaries with keys:
                        - 'center': (x, y) center point
                        - 'class_id': Class ID
                        - 'box': [x1, y1, x2, y2] bounding box
                        - 'confidence': Detection confidence

        Returns:
            Dictionary of tracked objects with ID as key
        """
        # If no detections, mark all objects as disappeared
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1

                # Deregister if object disappears for too many frames
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            return self.objects

        # If no existing objects, register all detections
        if len(self.objects) == 0:
            for det in detections:
                self.register(
                    det['center'],
                    det['class_id'],
                    det['box'],
                    det['confidence']
                )
        else:
            # Match existing objects to new detections
            object_ids = list(self.objects.keys())
            object_centroids = [self.objects[object_id]['centroid'] for object_id in object_ids]

            # Get centroids from detections
            detection_centroids = [det['center'] for det in detections]

            # Compute distances between each pair of object centroids and detection centroids
            D = self._compute_distances(object_centroids, detection_centroids)

            # Match objects to detections using Hungarian algorithm or simple greedy matching
            # For simplicity, we'll use a greedy approach here
            rows, cols = self._greedy_matching(D)

            # Keep track of which objects and detections we've processed
            used_rows = set()
            used_cols = set()

            # Update matched objects
            for (row, col) in zip(rows, cols):
                # Skip if distance is too large
                if D[row, col] > self.min_distance:
                    continue

                object_id = object_ids[row]
                det = detections[col]

                # Update object with new detection
                self.objects[object_id] = {
                    'centroid': det['center'],
                    'class_id': det['class_id'],
                    'bbox': det['box'],
                    'confidence': det['confidence']
                }
                self.disappeared[object_id] = 0
                self.tracks[object_id].append(det['center'])

                used_rows.add(row)
                used_cols.add(col)

            # Get unmatched rows (objects)
            unused_rows = set(range(D.shape[0])) - used_rows

            # Mark unmatched objects as disappeared
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1

                # Deregister if object disappears for too many frames
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # Register new detections
            unused_cols = set(range(D.shape[1])) - used_cols
            for col in unused_cols:
                det = detections[col]
                self.register(
                    det['center'],
                    det['class_id'],
                    det['box'],
                    det['confidence']
                )

        return self.objects

    def _compute_distances(self, centroids_a, centroids_b):
        """Compute distance matrix between two sets of centroids.

        Args:
            centroids_a: List of (x, y) tuples
            centroids_b: List of (x, y) tuples

        Returns:
            Distance matrix
        """
        # Convert to numpy arrays
        centroids_a = np.array(centroids_a)
        centroids_b = np.array(centroids_b)

        # Compute Euclidean distances
        D = np.zeros((len(centroids_a), len(centroids_b)))

        for i in range(len(centroids_a)):
            for j in range(len(centroids_b)):
                D[i, j] = np.sqrt((centroids_a[i][0] - centroids_b[j][0]) ** 2 +
                                  (centroids_a[i][1] - centroids_b[j][1]) ** 2)

        return D

    def _greedy_matching(self, D):
        """Simple greedy matching of distance matrix.

        Args:
            D: Distance matrix

        Returns:
            Matched row, col indices
        """
        # Make a copy of D
        D_copy = D.copy()

        rows = []
        cols = []

        while D_copy.size > 0:
            # Find minimum distance
            min_idx = np.argmin(D_copy)
            min_row, min_col = np.unravel_index(min_idx, D_copy.shape)

            rows.append(min_row)
            cols.append(min_col)

            # Remove used row and column
            D_copy = np.delete(D_copy, min_row, axis=0)
            if D_copy.size > 0:
                D_copy = np.delete(D_copy, min_col, axis=1)

            # Break if D is empty
            if D_copy.size == 0:
                break

        return rows, cols

    def draw_tracks(self, frame, max_points=50):
        """Draw tracking paths on frame.

        Args:
            frame: Image to draw on
            max_points: Maximum number of track points to draw

        Returns:
            Frame with tracks drawn
        """
        # Make a copy of the frame
        result = frame.copy()

        # Colors for different classes (BGR format)
        colors = {
            0: (0, 255, 0),  # person: green
            1: (255, 0, 0),  # bicycle: blue
            2: (0, 0, 255),  # car: red
            3: (255, 255, 0),  # motorcycle: cyan
            5: (255, 0, 255),  # bus: magenta
            7: (0, 255, 255)  # truck: yellow
        }

        # Draw tracks for each object
        for object_id, track in self.tracks.items():
            if object_id not in self.objects:
                continue  # Skip deregistered objects

            color = colors.get(self.objects[object_id]['class_id'], (0, 255, 0))

            # Draw limited track history
            points = track[-max_points:]
            for i in range(1, len(points)):
                cv2.line(result,
                         (points[i - 1][0], points[i - 1][1]),
                         (points[i][0], points[i][1]),
                         color, 2)

            # Draw ID next to the current position
            current_pos = track[-1]
            cv2.putText(result,
                        f"ID: {object_id}",
                        (current_pos[0] + 10, current_pos[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return result


# Example usage
if __name__ == "__main__":
    # Create a tracker
    tracker = ObjectTracker()

    # Test with simulated detections
    detections = [
        {'center': (100, 100), 'class_id': 2, 'box': [90, 90, 110, 110], 'confidence': 0.9},
        {'center': (200, 200), 'class_id': 0, 'box': [190, 190, 210, 210], 'confidence': 0.8}
    ]

    objects = tracker.update(detections)
    print(f"Tracking {len(objects)} objects")