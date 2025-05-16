# Urban Mobility Analytics Platform

A computer vision system for analyzing traffic patterns and urban mobility using object detection and tracking.

## Overview

This project demonstrates how computer vision can be applied to traffic analysis, creating a complete system that:

1. Detects vehicles and pedestrians in video footage
2. Tracks objects across frames to create movement trajectories 
3. Analyzes patterns such as traffic density and flow directions
4. Visualizes results through an interactive dashboard

## Features

- **Object Detection**: Uses YOLOv8 to identify vehicles, pedestrians, and other objects
- **Object Tracking**: Follows the same objects across multiple frames
- **Trajectory Analysis**: Studies the paths objects take through the scene
- **Traffic Flow Visualization**: Shows the dominant directions of movement
- **Interactive Dashboard**: Allows for exploration of analysis results

## Demo

The dashboard includes a demo button that processes a sample traffic video, displaying detection, tracking, and analysis results without requiring users to upload their own footage.

## Getting Started

### Installation

1. Clone the repository:
git clone https://github.com/yourusername/urban-mobility-analytics.git
cd urban-mobility-analytics

2. Create a virtual environment and install dependencies:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

3. Run the Streamlit app:
streamlit run dashboard/app.py

### Usage

1. Upload a traffic video (short clips of 10-30 seconds work best)
2. Choose processing options and click "Process Video"
3. Explore the analysis through the interactive visualizations

## Project Structure

- `dashboard/`: Streamlit application code
- `src/`: Core processing modules
- `detector.py`: Object detection using YOLOv8
- `tracker.py`: Object tracking implementation
- `traffic_analysis.py`: Traffic pattern analysis
- `trajectory_analysis.py`: Trajectory analysis tools
- `data/`: Data files and sample videos

## Technologies Used

- Python
- OpenCV
- PyTorch & YOLOv8
- Streamlit
- Pandas & NumPy
- Plotly & Matplotlib

## Deployment

This application can be deployed on Streamlit Cloud. See the deployment section for details.

## About This Project

This project was created as a portfolio piece to demonstrate computer vision and data analysis skills. It combines object detection, tracking, and interactive visualization to create a complete urban mobility analytics platform.
