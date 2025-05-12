import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import base64

# Set page config
st.set_page_config(
    page_title="Urban Mobility Analytics",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("Urban Mobility Analytics Dashboard")


# Load pre-computed demo data
@st.cache_data
def load_demo_data():
    """Load pre-computed tracking data for demo."""
    try:
        # Path to demo data included with the app
        demo_data_path = os.path.join(os.path.dirname(__file__), "demo", "sample_tracking.csv")
        if os.path.exists(demo_data_path):
            return pd.read_csv(demo_data_path)
        else:
            st.error(f"Demo data not found at: {demo_data_path}")
            return None
    except Exception as e:
        st.error(f"Error loading demo data: {e}")
        return None


# Display GIF function
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


# Show the demo
st.header("Urban Mobility Analytics Demo")
st.markdown("""
This is a demonstration of our Urban Mobility Analytics platform, which uses computer vision to:
- Detect vehicles, pedestrians, and other objects in traffic videos
- Track objects across frames to create movement trajectories
- Analyze patterns such as traffic density and flow directions
- Visualize results through this interactive dashboard

Below you'll see a pre-processed traffic video and the resulting analysis.
""")

# Display demo video/GIF
demo_gif_path = os.path.join(os.path.dirname(__file__), "demo", "sample_tracked.gif")
st.subheader("Traffic Video with Object Tracking")
display_success = display_gif(demo_gif_path)

if not display_success:
    st.warning("Demo visualization not available. Please check out the GitHub repository for full demo videos.")

# Load and display demo data
tracking_df = load_demo_data()

if tracking_df is not None:
    # Set up video dimensions
    video_width = 1280  # Default for demo
    video_height = 720  # Default for demo

    # Create visualization tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Trajectories", "Speed Analysis", "Flow Analysis"])

    # Define visualization functions and tab content here...
    # [Copy the visualization code from your existing app]
else:
    st.error("Could not load demo data. Please visit the GitHub repository for the full application.")

# Add footer with project info
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <h3>Urban Mobility Analytics Platform</h3>
    <p>A computer vision system for analyzing traffic patterns and urban mobility</p>
    <p><a href="https://github.com/yourusername/urban-mobility-analytics" target="_blank">View on GitHub for full functionality</a></p>
</div>
""", unsafe_allow_html=True)