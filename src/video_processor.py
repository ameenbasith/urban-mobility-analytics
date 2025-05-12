import cv2
import os
import numpy as np
from pathlib import Path


class VideoProcessor:
    def __init__(self, video_path):
        """Initialize the video processor with a video file path."""
        self.video_path = video_path
        self.cap = None
        self.frame_count = 0
        self.fps = 0
        self.width = 0
        self.height = 0

    def open_video(self):
        """Open the video file and get basic info."""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")

        # Get video properties
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Opened video with {self.frame_count} frames, {self.fps} FPS, "
              f"resolution: {self.width}x{self.height}")

    def extract_frames(self, output_dir, sample_rate=1):
        """Extract frames from video at given sample rate.

        Args:
            output_dir: Directory to save frames
            sample_rate: Save every Nth frame
        """
        if not self.cap or not self.cap.isOpened():
            self.open_video()

        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        count = 0
        saved = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            if count % sample_rate == 0:
                # Save frame as a JPG file
                output_file = output_path / f"frame_{saved:05d}.jpg"
                cv2.imwrite(str(output_file), frame)
                saved += 1

            count += 1

            if count % 100 == 0:
                print(f"Processed {count} frames, saved {saved} frames")

        print(f"Finished processing video. Saved {saved} frames to {output_dir}")
        self.cap.release()

    def close(self):
        """Release the video capture object."""
        if self.cap and self.cap.isOpened():
            self.cap.release()


# Example usage
if __name__ == "__main__":
    # Replace with the path to your video
    video_file = "../data/raw_videos/sample_traffic.mp4"
    output_directory = "../data/frames"

    processor = VideoProcessor(video_file)
    # Extract every 10th frame
    processor.extract_frames(output_directory, sample_rate=10)
    processor.close()