import cv2
import time
import os
from pathlib import Path
import numpy as np
from detector import ObjectDetector
import pandas as pd


def process_video(video_path, output_path, sample_rate=1, confidence=0.3):
    """Process a video with object detection and save results.

    Args:
        video_path: Path to input video
        output_path: Path to save processed video
        sample_rate: Process every Nth frame
        confidence: Detection confidence threshold

    Returns:
        DataFrame with detection data
    """
    # Initialize the detector
    detector = ObjectDetector()

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing video: {width}x{height}, {fps} FPS, {frame_count} frames")

    # Create video writer for output
    output_dir = os.path.dirname(output_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Storage for detection data
    detection_data = []

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

            # Draw results on the frame
            processed_frame = detector.draw_detections(frame, results)

            # Write frame to output video
            out.write(processed_frame)

            # Store detection data
            for det in results['detections']:
                detection_data.append({
                    'frame': frame_idx,
                    'class_id': det['class_id'],
                    'class_name': det['class_name'],
                    'confidence': det['confidence'],
                    'x1': det['box'][0],
                    'y1': det['box'][1],
                    'x2': det['box'][2],
                    'y2': det['box'][3],
                    'center_x': det['center'][0],
                    'center_y': det['center'][1],
                    'timestamp': frame_idx / fps
                })

            processed_frames += 1

            # Print progress every 10 processed frames
            if processed_frames % 10 == 0:
                elapsed = time.time() - start_time
                fps_processing = processed_frames / max(elapsed, 0.001)
                remaining_frames = (frame_count - frame_idx) / sample_rate
                eta = remaining_frames / max(fps_processing, 0.001)

                print(f"Processed {processed_frames} frames ({frame_idx}/{frame_count}), "
                      f"FPS: {fps_processing:.2f}, ETA: {eta:.1f}s")

        frame_idx += 1

    # Release resources
    cap.release()
    out.release()

    print(f"Video processing complete. Processed {processed_frames} frames, "
          f"found {len(detection_data)} objects.")
    print(f"Output saved to {output_path}")

    # Create DataFrame
    df = pd.DataFrame(detection_data)

    # Save detection data
    csv_path = output_path.replace('.mp4', '_detections.csv')
    df.to_csv(csv_path, index=False)
    print(f"Detection data saved to {csv_path}")

    return df


if __name__ == "__main__":
    # Replace with your video path
    video_file = "../data/raw_videos/sample_traffic.mp4"
    output_file = "../results/processed_video.mp4"

    # Process every 2nd frame with 30% confidence threshold
    detection_df = process_video(video_file, output_file, sample_rate=2, confidence=0.3)