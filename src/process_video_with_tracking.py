import cv2
import time
import os
from pathlib import Path
import numpy as np
import pandas as pd
from detector import ObjectDetector
from tracker import ObjectTracker


def process_video_with_tracking(video_path, output_path, sample_rate=1, confidence=0.3):
    """Process a video with object detection and tracking.

    Args:
        video_path: Path to input video
        output_path: Path to save processed video
        sample_rate: Process every Nth frame
        confidence: Detection confidence threshold

    Returns:
        DataFrame with tracking data
    """
    # Initialize detector and tracker
    detector = ObjectDetector()
    tracker = ObjectTracker(max_disappeared=10, min_distance=50)

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing video with tracking: {width}x{height}, {fps} FPS, {frame_count} frames")

    # Create video writer for output
    output_dir = os.path.dirname(output_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Storage for tracking data
    tracking_data = []

    # Process the video
    frame_idx = 0
    processed_frames = 0

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process every Nth frame
        if frame_idx % sample_rate == 0:
            # Run detection
            results = detector.detect(frame, confidence_threshold=confidence)

            # Update tracker
            objects = tracker.update(results['detections'])

            # Draw detections and tracks
            processed_frame = detector.draw_detections(frame, results)
            processed_frame = tracker.draw_tracks(processed_frame)

            # Write frame to output video
            out.write(processed_frame)

            # Store tracking data
            for object_id, obj in objects.items():
                tracking_data.append({
                    'frame': frame_idx,
                    'object_id': object_id,
                    'class_id': obj['class_id'],
                    'class_name': next((d['class_name'] for d in results['detections']
                                        if d['class_id'] == obj['class_id']), 'unknown'),
                    'confidence': obj['confidence'],
                    'x1': obj['bbox'][0],
                    'y1': obj['bbox'][1],
                    'x2': obj['bbox'][2],
                    'y2': obj['bbox'][3],
                    'center_x': obj['centroid'][0],
                    'center_y': obj['centroid'][1],
                    'timestamp': frame_idx / fps,
                    'track_length': len(tracker.tracks[object_id])
                })

            processed_frames += 1

            # Print progress every 10 processed frames
            if processed_frames % 10 == 0:
                elapsed = time.time() - start_time
                fps_processing = processed_frames / max(elapsed, 0.001)
                remaining_frames = (frame_count - frame_idx) / sample_rate
                eta = remaining_frames / max(fps_processing, 0.001)

                print(f"Processed {processed_frames} frames ({frame_idx}/{frame_count}), "
                      f"FPS: {fps_processing:.2f}, Tracking {len(objects)} objects, ETA: {eta:.1f}s")

        frame_idx += 1

    # Release resources
    cap.release()
    out.release()

    print(f"Video processing with tracking complete. Processed {processed_frames} frames, "
          f"tracked {tracker.next_object_id} objects.")
    print(f"Output saved to {output_path}")

    # Create DataFrame
    df = pd.DataFrame(tracking_data)

    # Save tracking data
    csv_path = output_path.replace('.mp4', '_tracking.csv')
    df.to_csv(csv_path, index=False)
    print(f"Tracking data saved to {csv_path}")

    return df


if __name__ == "__main__":
    # Replace with your video path
    video_file = "../data/raw_videos/sample_traffic.mp4"
    output_file = "../results/tracked_video.mp4"

    # Process every 2nd frame with 30% confidence threshold
    tracking_df = process_video_with_tracking(video_file, output_file, sample_rate=2, confidence=0.3)