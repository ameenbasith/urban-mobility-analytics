import streamlit as st
import pandas as pd
import numpy as np
import imageio
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import cv2
from pathlib import Path
import base64
from io import BytesIO  # Add this import if you don't have it already

# Create permanent directories for storing uploaded and processed videos
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploaded_videos")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed_videos")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from src.detector import ObjectDetector
from src.tracker import ObjectTracker
from src.traffic_analysis import TrafficAnalyzer
from src.trajectory_analysis import TrajectoryAnalyzer

# Set page config
st.set_page_config(
    page_title="Urban Mobility Analytics",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("Urban Mobility Analytics Dashboard")


def convert_video_to_gif(video_path, gif_path, fps=5, max_size=480):
    """Convert a video to an animated GIF with size and frame rate reduction."""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Could not open video: {video_path}")
        return None

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate new dimensions while maintaining aspect ratio
    if width > height:
        new_width = min(width, max_size)
        new_height = int(height * (new_width / width))
    else:
        new_height = min(height, max_size)
        new_width = int(width * (new_height / height))

    # Calculate frame sampling rate based on desired FPS
    sample_rate = max(1, int(video_fps / fps))

    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Converting video to GIF...")

    # Collect frames for the GIF
    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process every Nth frame
        if frame_idx % sample_rate == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize the frame
            pil_img = Image.fromarray(frame_rgb)
            pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)

            # Add to frames
            frames.append(np.array(pil_img))

        # Update progress
        if frame_idx % 20 == 0:
            progress = min(int(100 * frame_idx / frame_count), 100)
            progress_bar.progress(progress)
            status_text.text(f"Converting to GIF: {progress}%")

        frame_idx += 1

    # Release the video
    cap.release()

    # Save as GIF
    status_text.text("Saving GIF...")
    imageio.mimsave(gif_path, frames, fps=fps)

    # Complete
    progress_bar.progress(100)
    status_text.text("GIF conversion complete!")

    return gif_path


def display_gif(gif_path):
    """Display a GIF in Streamlit."""
    try:
        # Check if file exists
        if not os.path.exists(gif_path):
            st.error(f"GIF not found: {gif_path}")
            return False

        # Display the GIF using markdown
        with open(gif_path, "rb") as file:
            contents = file.read()
            data_url = base64.b64encode(contents).decode("utf-8")
            st.markdown(
                f'<img src="data:image/gif;base64,{data_url}" alt="processed video" width="100%">',
                unsafe_allow_html=True,
            )

        return True
    except Exception as e:
        st.error(f"Error displaying GIF: {e}")
        return False

# Add Demo Button
st.sidebar.markdown("---")
st.sidebar.subheader("Try a Demo")
st.sidebar.markdown("Don't have a video? Try our sample traffic video.")

if st.sidebar.button("Load and Process Demo"):
    # Path to sample video included with the app
    sample_video_path = os.path.join(os.path.dirname(__file__), "demo", "sample_traffic.mp4")

    # Check if sample video exists
    if not os.path.exists(sample_video_path):
        st.sidebar.error(f"Sample video not found. Please add a video at: {sample_video_path}")
    else:
        # Show the demo video
        st.subheader("Sample Traffic Video")
        st.video(sample_video_path)

        # Get video info
        cap = cv2.VideoCapture(sample_video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Set default processing parameters
        confidence = 0.3
        sample_rate = 2

        # Process the video with tracking
        st.subheader("Processing demo video with tracking...")

        # Output path for video and GIF
        processed_video_path = os.path.join(PROCESSED_DIR, "demo_tracked.mp4")
        processed_gif_path = os.path.join(PROCESSED_DIR, "demo_tracked.gif")

        # Initialize detector and tracker
        from src.detector import ObjectDetector
        from src.tracker import ObjectTracker

        detector = ObjectDetector()
        tracker = ObjectTracker(max_disappeared=10, min_distance=50)

        # Open the video
        cap = cv2.VideoCapture(sample_video_path)
        if not cap.isOpened():
            st.error(f"Could not open video: {sample_video_path}")
            st.stop()

        # Get video properties again (to be safe)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create video writer for output
        os.makedirs(os.path.dirname(processed_video_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))

        # Storage for tracking data
        tracking_data = []

        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Processing video with tracking...")

        # Reset tracker object ID counter at start
        tracker.next_object_id = 0

        # Process frames
        frame_idx = 0
        processed_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Update progress
            if frame_idx % 10 == 0:
                progress = min(int(100 * frame_idx / frame_count), 100)
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_idx}/{frame_count} ({progress}%)")

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
            else:
                # For non-processed frames, just write the original
                out.write(frame)

            frame_idx += 1

        # Release resources
        cap.release()
        out.release()

        # Complete the progress bar
        progress_bar.progress(100)
        status_text.text("Processing complete!")

        # Create DataFrame
        tracking_df = pd.DataFrame(tracking_data)

        if len(tracking_df) > 0:
            st.success(f"Demo video processed successfully! Tracked {tracking_df['object_id'].nunique()} objects.")

            # Convert processed video to GIF
            st.subheader("Converting processed video to GIF...")
            convert_video_to_gif(processed_video_path, processed_gif_path, fps=5, max_size=480)

            # Show processed video as GIF
            st.subheader("Processed Demo Video with Tracking")
            display_gif(processed_gif_path)

            # Save to session state for analysis
            st.session_state['tracking_df'] = tracking_df
            st.session_state['analysis_type'] = 'tracking'
            st.session_state['video_width'] = width
            st.session_state['video_height'] = height

            # Prompt to view analysis
            st.info("Demo processing complete! Use the tabs below to explore the analysis.")
        else:
            st.error("Demo processing failed or no objects were detected/tracked.")

# Function to resize a video to a smaller resolution
def resize_video(input_path, output_path, scale_factor=0.5):
    """Resize a video to a smaller resolution."""
    cap = cv2.VideoCapture(input_path)

    # Get original dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate new dimensions
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

    # Process the video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Resizing video...")

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame
        resized_frame = cv2.resize(frame, (new_width, new_height))

        # Write to output
        out.write(resized_frame)

        # Update progress every 10 frames
        if i % 10 == 0:
            progress = int(100 * i / frame_count)
            progress_bar.progress(progress)

    # Release resources
    cap.release()
    out.release()
    progress_bar.progress(100)
    status_text.text("Video resizing complete!")

    return output_path, new_width, new_height


# Simple video processing function (no chunking)
def process_video_with_detection(video_path, output_path, confidence=0.3, sample_rate=2):
    """Process a video with object detection."""
    # Initialize detector
    detector = ObjectDetector()

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Could not open video: {video_path}")
        return None

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create video writer for output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Storage for detection data
    detection_data = []

    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Processing video...")

    # Process frames
    frame_idx = 0
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Update progress
        if frame_idx % 10 == 0:
            progress = min(int(100 * frame_idx / frame_count), 100)
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_idx}/{frame_count} ({progress}%)")

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
        else:
            # For non-processed frames, just write the original
            out.write(frame)

        frame_idx += 1

    # Release resources
    cap.release()
    out.release()

    # Complete the progress bar
    progress_bar.progress(100)
    status_text.text("Processing complete!")

    # Create DataFrame
    df = pd.DataFrame(detection_data)

    return df


# Simple tracking function (no chunking)
def process_video_with_tracking(video_path, output_path, confidence=0.3, sample_rate=2):
    """Process a video with object detection and tracking."""
    # Initialize detector and tracker
    detector = ObjectDetector()
    tracker = ObjectTracker(max_disappeared=10, min_distance=50)

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Could not open video: {video_path}")
        return None

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create video writer for output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Storage for tracking data
    tracking_data = []

    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Processing video with tracking...")

    # Reset tracker object ID counter at start
    tracker.next_object_id = 0

    # Process frames
    frame_idx = 0
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Update progress
        if frame_idx % 10 == 0:
            progress = min(int(100 * frame_idx / frame_count), 100)
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_idx}/{frame_count} ({progress}%)")

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
        else:
            # For non-processed frames, just write the original
            out.write(frame)

        frame_idx += 1

    # Release resources
    cap.release()
    out.release()

    # Complete the progress bar
    progress_bar.progress(100)
    status_text.text("Processing complete!")

    # Create DataFrame
    df = pd.DataFrame(tracking_data)

    return df

def convert_video_to_gif(video_path, gif_path, fps=5, max_size=480):
    """Convert a video to an animated GIF with size and frame rate reduction."""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Could not open video: {video_path}")
        return None

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate new dimensions while maintaining aspect ratio
    if width > height:
        new_width = min(width, max_size)
        new_height = int(height * (new_width / width))
    else:
        new_height = min(height, max_size)
        new_width = int(width * (new_height / height))

    # Calculate frame sampling rate based on desired FPS
    sample_rate = max(1, int(video_fps / fps))

    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Converting video to GIF...")

    # Collect frames for the GIF
    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process every Nth frame
        if frame_idx % sample_rate == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize the frame
            pil_img = Image.fromarray(frame_rgb)
            pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)

            # Add to frames
            frames.append(np.array(pil_img))

        # Update progress
        if frame_idx % 20 == 0:
            progress = min(int(100 * frame_idx / frame_count), 100)
            progress_bar.progress(progress)
            status_text.text(f"Converting to GIF: {progress}%")

        frame_idx += 1

    # Release the video
    cap.release()

    # Save as GIF
    status_text.text("Saving GIF...")
    imageio.mimsave(gif_path, frames, fps=fps)

    # Complete
    progress_bar.progress(100)
    status_text.text("GIF conversion complete!")

    return gif_path


def display_gif(gif_path):
    """Display a GIF in Streamlit."""
    try:
        # Check if file exists
        if not os.path.exists(gif_path):
            st.error(f"GIF not found: {gif_path}")
            return False

        # Display the GIF using markdown
        with open(gif_path, "rb") as file:
            contents = file.read()
            data_url = base64.b64encode(contents).decode("utf-8")
            st.markdown(
                f'<img src="data:image/gif;base64,{data_url}" alt="processed video" width="100%">',
                unsafe_allow_html=True,
            )

        # Also provide a download link
        with open(gif_path, "rb") as file:
            btn = st.download_button(
                label="Download GIF",
                data=file,
                file_name=os.path.basename(gif_path),
                mime="image/gif"
            )

        return True
    except Exception as e:
        st.error(f"Error displaying GIF: {e}")
        return False

# Create a permanent directory for storing uploaded and processed videos
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploaded_videos")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_videos")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Sidebar
st.sidebar.header("Controls")

# Data source selection
data_option = st.sidebar.radio(
    "Choose data source",
    ["Upload video for processing", "Upload pre-processed data"]
)

if data_option == "Upload video for processing":
    # Video upload with clear guidance
    st.sidebar.info(
        "⚠️ Please upload a short video (under 20 seconds). Large videos may cause memory issues. For best results, use videos showing traffic or pedestrians.")
    uploaded_video = st.sidebar.file_uploader("Upload traffic video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        # Save the uploaded video to a file
        video_filename = uploaded_video.name
        original_path = os.path.join(UPLOAD_DIR, video_filename)

        with open(original_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        # Show the uploaded video
        st.subheader("Uploaded Video")

        # Get video info
        cap = cv2.VideoCapture(original_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        cap.release()

        col1, col2 = st.columns([3, 1])

        with col1:
            # Show video
            st.video(original_path)

        with col2:
            st.write("**Video Info:**")
            st.write(f"Resolution: {width}x{height}")
            st.write(f"Duration: {duration:.2f} seconds")
            st.write(f"Frames: {frame_count}")
            st.write(f"FPS: {fps:.2f}")

            # Option to resize
            resize_option = st.checkbox("Resize video", value=True,
                                        help="Resize the video to a smaller resolution to reduce memory usage.")

            if resize_option:
                scale_factor = st.slider(
                    "Resize scale factor",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                    help="A value of 0.5 reduces the video to half its original width and height (25% of original size)."
                )

        # Processing options
        st.sidebar.subheader("Processing Options")

        # Confidence threshold with explanation
        confidence = st.sidebar.slider(
            "Detection Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.3,
            step=0.1,
            help="Only objects detected with confidence above this threshold will be included. Higher values reduce false positives but may miss some objects."
        )

        # Sample rate with explanation
        sample_rate = st.sidebar.slider(
            "Process every Nth frame",
            min_value=1,
            max_value=10,
            value=2,
            step=1,
            help="Skip frames to speed up processing. A value of 2 means every second frame is processed, 3 means every third frame, etc."
        )

        # Select processing type with explanation
        process_type = st.sidebar.radio(
            "Select processing type",
            ["Detection only", "Detection with tracking"],
            help="Detection identifies objects in each frame. Tracking follows objects across frames to analyze their movement."
        )

        # Process button
        if st.sidebar.button("Process Video"):
            # Resize if requested
            video_to_process = original_path
            if resize_option:
                st.subheader("Resizing video...")
                resized_path = os.path.join(UPLOAD_DIR, f"resized_{video_filename}")
                video_to_process, resized_width, resized_height = resize_video(
                    original_path, resized_path, scale_factor
                )
                st.success(f"Video resized to {resized_width}x{resized_height}")

            # Process the video
            if process_type == "Detection only":
                st.subheader("Processing video with object detection...")

                # Output path
                processed_video_path = os.path.join(PROCESSED_DIR, f"processed_{os.path.basename(video_to_process)}")

                # Process the video
                detection_df = process_video_with_detection(
                    video_to_process,
                    processed_video_path,
                    confidence,
                    sample_rate
                )

                if detection_df is not None and len(detection_df) > 0:
                    st.success(f"Video processed successfully! Detected {len(detection_df)} objects.")

                    # Convert video to GIF
                    st.subheader("Converting processed video to GIF for display...")
                    gif_path = os.path.join(PROCESSED_DIR, f"processed_{os.path.basename(video_to_process)}.gif")
                    convert_video_to_gif(processed_video_path, gif_path, fps=5, max_size=480)

                    # Show processed video as GIF
                    st.subheader("Processed Video with Detections")
                    display_gif(gif_path)

                    # Save to session state for analysis
                    st.session_state['detection_df'] = detection_df
                    st.session_state['analysis_type'] = 'detection'

                    # Get dimensions from processed video
                    cap = cv2.VideoCapture(processed_video_path)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()

                    st.session_state['video_width'] = width
                    st.session_state['video_height'] = height

                    # Prompt to view analysis
                    st.info("Video processing complete! Use the tabs below to explore the analysis.")
                else:
                    st.error("Processing failed or no objects were detected.")

            else:  # Detection with tracking
                st.subheader("Processing video with detection and tracking...")

                # Output path
                processed_video_path = os.path.join(PROCESSED_DIR, f"tracked_{os.path.basename(video_to_process)}")

                # Process the video
                tracking_df = process_video_with_tracking(
                    video_to_process,
                    processed_video_path,
                    confidence,
                    sample_rate
                )

                if tracking_df is not None and len(tracking_df) > 0:
                    st.success(f"Video processed successfully! Tracked {tracking_df['object_id'].nunique()} objects.")

                    # Convert video to GIF
                    st.subheader("Converting processed video to GIF for display...")
                    gif_path = os.path.join(PROCESSED_DIR, f"tracked_{os.path.basename(video_to_process)}.gif")
                    convert_video_to_gif(processed_video_path, gif_path, fps=5, max_size=480)

                    # Show processed video as GIF
                    st.subheader("Processed Video with Tracking")
                    display_gif(gif_path)

                    # Save to session state for analysis
                    st.session_state['tracking_df'] = tracking_df
                    st.session_state['analysis_type'] = 'tracking'

                    # Get dimensions from processed video
                    cap = cv2.VideoCapture(processed_video_path)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()

                    st.session_state['video_width'] = width
                    st.session_state['video_height'] = height

                    # Prompt to view analysis
                    st.info("Video processing complete! Use the tabs below to explore the analysis.")
                else:
                    st.error("Processing failed or no objects were detected/tracked.")

    else:
        st.info("Please upload a video file for processing.")

else:  # Upload pre-processed data
    # Data type selection
    data_type = st.sidebar.radio(
        "Select data type",
        ["Detection data", "Tracking data"]
    )

    # File uploader
    uploaded_file = st.sidebar.file_uploader(f"Upload {data_type} CSV", type=["csv"])

    if uploaded_file is not None:
        if data_type == "Detection data":
            df = pd.read_csv(uploaded_file)
            st.session_state['detection_df'] = df
            st.session_state['analysis_type'] = 'detection'

            # Default video dimensions if not known
            st.session_state['video_width'] = 1920
            st.session_state['video_height'] = 1080

            st.success(f"Loaded {len(df)} detection records. Use the tabs below to explore the analysis.")
        else:
            df = pd.read_csv(uploaded_file)
            st.session_state['tracking_df'] = df
            st.session_state['analysis_type'] = 'tracking'

            # Default video dimensions if not known
            st.session_state['video_width'] = 1920
            st.session_state['video_height'] = 1080

            st.success(f"Loaded {len(df)} tracking records. Use the tabs below to explore the analysis.")
    else:
        st.info(f"Please upload a {data_type} CSV file.")

# Check if we have data to analyze
if ('detection_df' in st.session_state and st.session_state['analysis_type'] == 'detection') or \
        ('tracking_df' in st.session_state and st.session_state['analysis_type'] == 'tracking'):

    # Get the data and video dimensions
    if st.session_state['analysis_type'] == 'detection':
        df = st.session_state['detection_df']
        analyzer = TrafficAnalyzer(df)

        # Main tabs for detection data
        tab1, tab2, tab3 = st.tabs(["Overview", "Time Analysis", "Spatial Analysis"])

        with tab1:
            st.header("Traffic Overview")

            # Basic stats
            col1, col2, col3 = st.columns(3)

            with col1:
                total_objects = len(df)
                st.metric("Total Objects Detected", total_objects)
                st.caption("Total number of objects detected across all frames")

            with col2:
                unique_frames = df['frame'].nunique()
                st.metric("Frames Analyzed", unique_frames)
                st.caption("Number of video frames that were processed")

            with col3:
                class_count = df['class_name'].nunique()
                st.metric("Object Types", class_count)
                st.caption("Number of different object categories detected")

            # Object counts by class
            st.subheader("Object Distribution by Type")
            st.caption("This chart shows the total count of each type of object detected in the video")

            counts = analyzer.count_by_class()

            fig = px.bar(
                x=counts.index,
                y=counts.values,
                labels={'x': 'Object Type', 'y': 'Count'},
                color=counts.index,
                color_discrete_sequence=px.colors.qualitative.Bold
            )

            fig.update_layout(
                xaxis_title="Object Type",
                yaxis_title="Number of Objects",
                height=400,
            )

            st.plotly_chart(fig, use_container_width=True)

            # Show sample data
            st.subheader("Sample Detection Data")
            st.caption(
                "The first 10 rows of the detection data, showing frame numbers, object types, positions, and confidence scores")
            st.dataframe(df.head(10))

        with tab2:
            st.header("Time Analysis")
            st.caption("Analyze how the number and types of objects change over time in the video")

            # Time window selection
            time_window = st.slider(
                "Time Window (seconds)",
                min_value=1,
                max_value=30,
                value=5,
                step=1,
                help="Group data into time windows of this duration. Adjust to see trends at different time scales."
            )

            # Get counts over time
            time_counts = analyzer.count_over_time(time_window=time_window)

            # Convert the time_window index to strings for plotting
            time_counts_reset = time_counts.reset_index()
            time_counts_reset['time_window'] = time_counts_reset['time_window'].astype(str)

            # Melt the DataFrame for easier plotting
            melted_counts = pd.melt(
                time_counts_reset,
                id_vars=['time_window'],
                value_vars=[c for c in time_counts.columns if c != 'total'],
                var_name='Class',
                value_name='Count'
            )

            # Line chart
            st.subheader(f"Object Counts Over Time (Window: {time_window}s)")
            st.caption("This chart shows how the number of each type of object changes over time")

            fig = px.line(
                melted_counts,
                x='time_window',
                y='Count',
                color='Class',
                markers=True,
                line_shape='linear'
            )

            fig.update_layout(
                xaxis_title="Time Window",
                yaxis_title="Number of Objects",
                height=500,
                legend_title="Object Type"
            )

            st.plotly_chart(fig, use_container_width=True)

            # Stacked area chart
            st.subheader("Cumulative Object Counts")
            st.caption(
                "This chart shows the total number of objects over time, with different colors representing different object types")

            fig = px.area(
                melted_counts,
                x='time_window',
                y='Count',
                color='Class',
                line_shape='linear'
            )

            fig.update_layout(
                xaxis_title="Time Window",
                yaxis_title="Number of Objects",
                height=500,
                legend_title="Object Type"
            )

            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.header("Spatial Analysis")
            st.caption("Analyze where objects are located in the video frame")

            # Heatmap resolution
            grid_size = st.slider(
                "Heatmap Resolution",
                min_value=10,
                max_value=100,
                value=32,
                step=2,
                help="Number of grid cells in each dimension. Higher values give more detailed heatmaps."
            )

            # Class filter
            all_classes = sorted(df['class_name'].unique())
            selected_classes = st.multiselect(
                "Filter by object type",
                options=all_classes,
                default=all_classes,
                help="Select which types of objects to include in the analysis."
            )

            if not selected_classes:
                st.warning("Please select at least one object type.")
            else:
                # Filter data
                filtered_df = df[df['class_name'].isin(selected_classes)]

                # Generate heatmap
                st.subheader("Object Location Heatmap")
                st.caption(
                    "This heatmap shows where objects appear most frequently in the video frame. Brighter colors indicate higher object density.")

                # Get video dimensions
                video_width = st.session_state['video_width']
                video_height = st.session_state['video_height']

                # Create grid
                x_bins = np.linspace(0, video_width, grid_size + 1)
                y_bins = np.linspace(0, video_height, grid_size + 1)

                # Create 2D histogram using object centers
                heatmap, x_edges, y_edges = np.histogram2d(
                    filtered_df['center_x'],
                    filtered_df['center_y'],
                    bins=[x_bins, y_bins]
                )

                # Create heatmap with plotly
                fig = go.Figure(data=go.Heatmap(
                    z=heatmap.T,
                    x=[f"{x:.0f}" for x in x_bins[:-1]],
                    y=[f"{y:.0f}" for y in y_bins[:-1]],
                    colorscale='Viridis',
                    hoverongaps=False
                ))

                fig.update_layout(
                    xaxis_title="X Position (pixels)",
                    yaxis_title="Y Position (pixels)",
                    height=600,
                    title=f"Heatmap of {', '.join(selected_classes)} Locations"
                )

                st.plotly_chart(fig, use_container_width=True)

                # Movement paths visualization
                st.subheader("Object Positions")
                st.caption(
                    "This scatter plot shows the position of each detected object. Each point represents one detection.")

                # Scatter plot of all positions
                fig = px.scatter(
                    filtered_df,
                    x='center_x',
                    y='center_y',
                    color='class_name',
                    opacity=0.7,
                    hover_data=['class_name', 'confidence', 'frame'],
                    labels={'center_x': 'X Position (pixels)', 'center_y': 'Y Position (pixels)'},
                    title="All Detected Objects"
                )

                fig.update_layout(
                    xaxis=dict(range=[0, video_width]),
                    yaxis=dict(range=[0, video_height], autorange="reversed"),  # Reverse Y to match image coordinates
                    height=600,
                )

                st.plotly_chart(fig, use_container_width=True)

    else:  # Tracking data analysis
        df = st.session_state['tracking_df']
        trajectory_analyzer = TrajectoryAnalyzer(df)

        # Main tabs for tracking data
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Trajectories", "Speed Analysis", "Flow Analysis"])

        with tab1:
            st.header("Tracking Overview")
            st.caption("Summary of objects tracked across multiple frames")

            # Basic stats
            col1, col2, col3 = st.columns(3)

            with col1:
                total_records = len(df)
                st.metric("Total Tracking Records", total_records)
                st.caption("Total number of object positions recorded")

            with col2:
                unique_objects = df['object_id'].nunique()
                st.metric("Unique Objects Tracked", unique_objects)
                st.caption("Number of distinct objects that were tracked")

            with col3:
                unique_frames = df['frame'].nunique()
                st.metric("Frames Analyzed", unique_frames)
                st.caption("Number of video frames that were processed")

            # Get trajectory metrics
            metrics = trajectory_analyzer.analyze_trajectories()

            # Trajectory length distribution
            st.subheader("Trajectory Length Distribution")
            st.caption(
                "This histogram shows how many frames each object appears in. Longer trajectories indicate objects that were visible for more time.")

            fig = px.histogram(
                metrics,
                x='frames',
                color='class_name',
                nbins=30,
                opacity=0.7,
                barmode='overlay',
                labels={'frames': 'Number of Frames', 'class_name': 'Object Type'},
                title="Distribution of Trajectory Lengths"
            )

            fig.update_layout(
                xaxis_title="Trajectory Length (frames)",
                yaxis_title="Number of Objects",
                height=400,
                legend_title="Object Type"
            )

            st.plotly_chart(fig, use_container_width=True)

            # Show sample tracking data
            st.subheader("Sample Tracking Data")
            st.caption("The first 10 rows of tracking data, showing object IDs, positions, and frame numbers")
            st.dataframe(df.head(10))

        with tab2:
            st.header("Object Trajectories")
            st.caption("Visualize the paths that objects take through the scene")

            # Trajectory filtering
            min_frames = st.slider(
                "Minimum Trajectory Length (frames)",
                min_value=2,
                max_value=50,
                value=5,
                step=1,
                help="Only show trajectories that appear in at least this many frames. Higher values filter out brief appearances."
            )

            # Class filter
            all_classes = sorted(df['class_name'].unique())
            selected_classes = st.multiselect(
                "Filter by object type",
                options=all_classes,
                default=all_classes,
                help="Select which types of objects to include in the analysis."
            )

            if not selected_classes:
                st.warning("Please select at least one object type.")
            else:
                # Generate trajectory map
                st.subheader("Object Trajectory Map")
                st.caption(
                    "This map shows the paths that objects take through the video. Each line represents one object's movement trajectory.")

                fig = trajectory_analyzer.plot_trajectory_map(
                    class_filter=selected_classes,
                    min_frames=min_frames
                )

                st.plotly_chart(fig, use_container_width=True)

                # Trajectory metrics table
                st.subheader("Trajectory Metrics")
                st.caption(
                    "This table shows detailed metrics for each tracked object, including distance traveled, speed, and path straightness")

                # Filter metrics
                metrics = trajectory_analyzer.analyze_trajectories()
                filtered_metrics = metrics[
                    (metrics['class_name'].isin(selected_classes)) &
                    (metrics['frames'] >= min_frames)
                    ]

                if len(filtered_metrics) > 0:
                    # Display metrics table
                    display_cols = [
                        'object_id', 'class_name', 'frames', 'time_span',
                        'total_distance', 'displacement', 'avg_speed', 'straightness'
                    ]

                    # Add tooltips for each column
                    st.markdown("""
                    **Column Explanations:**
                    - **object_id**: Unique identifier for each tracked object
                    - **class_name**: Type of object (car, person, etc.)
                    - **frames**: Number of frames the object appears in
                    - **time_span**: Duration the object is visible (seconds)
                    - **total_distance**: Total path length traveled (pixels)
                    - **displacement**: Straight-line distance from start to end (pixels)
                    - **avg_speed**: Average speed (pixels/second)
                    - **straightness**: Ratio of displacement to total distance (0-1, higher is straighter)
                    """)

                    st.dataframe(
                        filtered_metrics[display_cols].sort_values('avg_speed', ascending=False),
                        hide_index=True
                    )
                else:
                    st.info("No trajectories match the current filters.")

        with tab3:
            st.header("Speed Analysis")
            st.caption("Analyze how fast different objects move through the scene")

            # Get trajectory metrics
            metrics = trajectory_analyzer.analyze_trajectories()

            # Filter out unrealistic speeds (likely tracking errors)
            q99 = metrics['avg_speed'].quantile(0.99)
            metrics = metrics[metrics['avg_speed'] <= q99]

            # Min trajectory length filter
            min_frames = st.slider(
                "Minimum Trajectory Length (frames)",
                min_value=2,
                max_value=50,
                value=5,
                step=1,
                key="speed_min_frames",
                help="Only include objects tracked for at least this many frames. Filters out brief appearances."
            )

            # Filter metrics
            metrics = metrics[metrics['frames'] >= min_frames]

            if len(metrics) > 0:
                # Speed distribution by class
                st.subheader("Speed Distribution by Object Type")
                st.caption(
                    "This box plot shows the distribution of speeds for each object type. The box shows the middle 50% of values, the line shows the median, and points show individual objects.")

                fig = px.box(
                    metrics,
                    x='class_name',
                    y='avg_speed',
                    color='class_name',
                    points="all",
                    labels={'class_name': 'Object Type', 'avg_speed': 'Average Speed (pixels/second)'},
                    title="Speed Distribution by Object Type"
                )

                fig.update_layout(
                    xaxis_title="Object Type",
                    yaxis_title="Average Speed (pixels/second)",
                    height=500,
                    showlegend=False
                )

                st.plotly_chart(fig, use_container_width=True)

                # Speed vs. straightness scatter plot
                st.subheader("Speed vs. Path Straightness")
                st.caption(
                    "This scatter plot shows the relationship between an object's speed and how straight its path is. A straightness of 1.0 means a perfectly straight line.")

                fig = px.scatter(
                    metrics,
                    x='straightness',
                    y='avg_speed',
                    color='class_name',
                    hover_data=['object_id', 'frames', 'total_distance'],
                    opacity=0.7,
                    labels={
                        'straightness': 'Path Straightness (0-1)',
                        'avg_speed': 'Average Speed (pixels/second)',
                        'class_name': 'Object Type'
                    },
                    title="Speed vs. Path Straightness by Object Type"
                )

                fig.update_layout(
                    xaxis_title="Path Straightness (0-1)",
                    yaxis_title="Average Speed (pixels/second)",
                    height=500,
                    legend_title="Object Type"
                )

                st.plotly_chart(fig, use_container_width=True)

                # Add explanation text
                st.markdown("""
                **Understanding this chart:**
                - Each point represents one tracked object
                - **Path Straightness** measures how direct an object's path is:
                  - 1.0 = perfectly straight line
                  - Lower values indicate more winding paths
                - **Average Speed** is measured in pixels per second
                - Looking for patterns can reveal interesting behaviors:
                  - Fast, straight objects might be vehicles on a main road
                  - Slow, winding objects might be pedestrians or vehicles navigating complex areas
                """)
            else:
                st.info("No trajectories match the current filters.")

        with tab4:
            st.header("Traffic Flow Analysis")
            st.caption("Visualize the dominant directions of movement in different areas of the scene")

            # Grid size selection
            grid_size = st.slider(
                "Flow Grid Resolution",
                min_value=4,
                max_value=16,
                value=8,
                step=1,
                help="Number of grid cells in each dimension. Higher values show more detailed flow patterns, lower values show broader trends."
            )

            # Min trajectory length filter
            min_frames = st.slider(
                "Minimum Trajectory Length (frames)",
                min_value=2,
                max_value=50,
                value=5,
                step=1,
                key="flow_min_frames",
                help="Only include objects tracked for at least this many frames. Filters out brief appearances."
            )

            # Generate flow analysis
            st.subheader("Traffic Flow Patterns")
            st.caption(
                "This visualization shows traffic flow patterns. The colors represent traffic density (brighter colors = more traffic), and the arrows show the average direction of movement in each grid cell.")

            # Filter data first
            filtered_data = df.copy()
            object_counts = filtered_data.groupby('object_id').size()
            valid_objects = object_counts[object_counts >= min_frames].index
            filtered_data = filtered_data[filtered_data['object_id'].isin(valid_objects)]

            # Create temporary analyzer with filtered data
            temp_analyzer = TrajectoryAnalyzer(filtered_data)

            # Generate flow analysis
            flow_grid, fig = temp_analyzer.flow_analysis(grid_size=grid_size)

            st.plotly_chart(fig, use_container_width=True)

            # Add detailed explanation
            st.markdown("""
            **How to interpret this visualization:**

            - **Colors**: The background heatmap shows traffic density - brighter/warmer colors indicate more objects passed through that area
            - **Arrows**: Each arrow shows the average direction of movement in that grid cell
              - Arrow length indicates how consistent the movement direction is
              - Shorter arrows mean objects move in many different directions in that area
              - Longer arrows indicate a strong, consistent flow direction

            This visualization can help identify:
            - Main traffic flows and their directions
            - Intersections where traffic flows meet
            - Areas with chaotic movement (short arrows in many directions)
            - Clear travel lanes (consistent arrow directions)
            """)

else:
    st.info("Please upload a video or pre-processed data to begin analysis.")

# Add footer with project info
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <h3>Urban Mobility Analytics Platform</h3>
    <p>A computer vision system for analyzing traffic patterns and urban mobility</p>
    <p>This portfolio project demonstrates object detection, tracking, and trajectory analysis techniques.</p>
    <p><small>Built with Python, OpenCV, YOLOv8, and Streamlit.</small></p>
</div>
""", unsafe_allow_html=True)