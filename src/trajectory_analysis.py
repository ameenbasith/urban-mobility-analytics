import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import os
import warnings


class TrajectoryAnalyzer:
    def __init__(self, tracking_data):
        """Initialize with tracking data DataFrame or CSV path.

        Args:
            tracking_data: DataFrame or path to CSV with tracking data
        """
        if isinstance(tracking_data, str):
            self.data = pd.read_csv(tracking_data)
        else:
            self.data = tracking_data.copy()

        print(f"Loaded tracking data with {len(self.data)} records")

        # Check if we have required columns
        required_columns = ['frame', 'object_id', 'class_name', 'center_x', 'center_y', 'timestamp']
        for col in required_columns:
            if col not in self.data.columns:
                print(f"Warning: Missing required column '{col}'")

    def analyze_trajectories(self):
        """Analyze object trajectories.

        Returns:
            DataFrame with trajectory metrics
        """
        # Group by object_id
        grouped = self.data.groupby('object_id')

        # Calculate trajectory metrics
        trajectory_metrics = []

        for object_id, group in grouped:
            # Sort by frame
            group = group.sort_values('frame')

            # Skip if too few points
            if len(group) < 2:
                continue

            # Calculate total distance
            total_distance = 0
            for i in range(1, len(group)):
                dx = group.iloc[i]['center_x'] - group.iloc[i - 1]['center_x']
                dy = group.iloc[i]['center_y'] - group.iloc[i - 1]['center_y']
                distance = np.sqrt(dx ** 2 + dy ** 2)
                total_distance += distance

            # Calculate displacement (straight-line distance)
            start_x, start_y = group.iloc[0]['center_x'], group.iloc[0]['center_y']
            end_x, end_y = group.iloc[-1]['center_x'], group.iloc[-1]['center_y']
            displacement = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)

            # Calculate average speed (pixels per second)
            time_span = group.iloc[-1]['timestamp'] - group.iloc[0]['timestamp']
            avg_speed = total_distance / max(time_span, 0.001)

            # Calculate straightness ratio
            straightness = displacement / max(total_distance, 0.001)

            # Get object class
            class_name = group.iloc[0]['class_name']

            # Store metrics
            trajectory_metrics.append({
                'object_id': object_id,
                'class_name': class_name,
                'frames': len(group),
                'time_span': time_span,
                'total_distance': total_distance,
                'displacement': displacement,
                'avg_speed': avg_speed,
                'straightness': straightness,
                'start_x': start_x,
                'start_y': start_y,
                'end_x': end_x,
                'end_y': end_y
            })

        # Create DataFrame
        metrics_df = pd.DataFrame(trajectory_metrics)

        return metrics_df

    def plot_trajectory_map(self, class_filter=None, min_frames=5, output_path=None):
        """Plot a map of all trajectories.

        Args:
            class_filter: List of classes to include (None for all)
            min_frames: Minimum number of frames for a trajectory
            output_path: Path to save the plot (optional)

        Returns:
            Plotly figure
        """
        # Filter data
        data = self.data.copy()

        if class_filter:
            data = data[data['class_name'].isin(class_filter)]

        # Get object IDs with enough frames
        object_counts = data['object_id'].value_counts()
        valid_objects = object_counts[object_counts >= min_frames].index
        data = data[data['object_id'].isin(valid_objects)]

        # Create figure
        fig = go.Figure()

        # Add trajectories
        for object_id, group in data.groupby('object_id'):
            # Sort by frame
            group = group.sort_values('frame')

            # Get class for color
            class_name = group.iloc[0]['class_name']

            # Add line
            fig.add_trace(go.Scatter(
                x=group['center_x'],
                y=group['center_y'],
                mode='lines+markers',
                name=f"{class_name} #{object_id}",
                line=dict(width=2),
                marker=dict(size=8),
                opacity=0.7,
                hoverinfo='text',
                text=[f"Frame: {row['frame']}<br>Object: {class_name} #{object_id}" for _, row in group.iterrows()]
            ))

        # Update layout
        fig.update_layout(
            title="Object Trajectories",
            xaxis_title="X Position",
            yaxis_title="Y Position",
            yaxis=dict(autorange="reversed"),  # Reverse Y to match image coordinates
            height=800,
            width=1000,
            showlegend=True,
            legend_title="Objects"
        )

        # Save if requested
        if output_path:
            try:
                # Change file extension to html if it's not already
                html_path = output_path.replace('.png', '.html')
                fig.write_html(html_path)
                print(f"Trajectory map saved to {html_path}")
            except Exception as e:
                warnings.warn(f"Could not save trajectory map to file: {e}")
                print(f"Warning: Could not save trajectory map to file: {e}")
                print("Continuing with analysis...")

        return fig

    def plot_speed_distributions(self, output_path=None):
        """Plot speed distributions by class.

        Args:
            output_path: Path to save the plot (optional)

        Returns:
            Matplotlib figure
        """
        # Get trajectory metrics
        metrics = self.analyze_trajectories()

        # Filter out unrealistic speeds (likely tracking errors)
        q99 = metrics['avg_speed'].quantile(0.99)
        metrics = metrics[metrics['avg_speed'] <= q99]

        # Create figure
        plt.figure(figsize=(12, 8))

        # Plot speed distributions
        sns.boxplot(x='class_name', y='avg_speed', data=metrics)
        sns.stripplot(x='class_name', y='avg_speed', data=metrics,
                      size=4, color=".3", linewidth=0, alpha=0.5)

        # Add labels and title
        plt.title('Speed Distribution by Object Class')
        plt.xlabel('Object Class')
        plt.ylabel('Average Speed (pixels/second)')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save if requested
        if output_path:
            try:
                plt.savefig(output_path)
                print(f"Speed distribution plot saved to {output_path}")
            except Exception as e:
                warnings.warn(f"Could not save speed distribution plot to file: {e}")
                print(f"Warning: Could not save speed distribution plot to file: {e}")
                print("Continuing with analysis...")

        return plt.gcf()

    def flow_analysis(self, grid_size=8, output_path=None):
        """Analyze traffic flow directions on a grid.

        Args:
            grid_size: Size of grid (NxN)
            output_path: Path to save the plot (optional)

        Returns:
            Tuple of (flow_grid, plotly figure)
        """
        # Get trajectory metrics
        metrics = self.analyze_trajectories()

        # Filter for reasonable trajectories
        metrics = metrics[metrics['frames'] >= 5]

        # Calculate direction vectors
        metrics['dir_x'] = metrics['end_x'] - metrics['start_x']
        metrics['dir_y'] = metrics['end_y'] - metrics['start_y']

        # Normalize directions
        norms = np.sqrt(metrics['dir_x'] ** 2 + metrics['dir_y'] ** 2)
        metrics['norm_dir_x'] = metrics['dir_x'] / np.maximum(norms, 0.001)
        metrics['norm_dir_y'] = metrics['dir_y'] / np.maximum(norms, 0.001)

        # Get image dimensions from data
        img_width = max(self.data['center_x'].max(), 1)
        img_height = max(self.data['center_y'].max(), 1)

        # Create grid
        cell_width = img_width / grid_size
        cell_height = img_height / grid_size

        # Initialize flow grid
        flow_grid = np.zeros((grid_size, grid_size, 2))  # [dx, dy] for each cell
        counts_grid = np.zeros((grid_size, grid_size))

        # Assign trajectories to grid cells based on start position
        for _, row in metrics.iterrows():
            # Determine grid cell for start position
            grid_x = min(int(row['start_x'] / cell_width), grid_size - 1)
            grid_y = min(int(row['start_y'] / cell_height), grid_size - 1)

            # Add normalized direction to grid cell
            flow_grid[grid_y, grid_x, 0] += row['norm_dir_x']
            flow_grid[grid_y, grid_x, 1] += row['norm_dir_y']
            counts_grid[grid_y, grid_x] += 1

        # Normalize flow vectors by count
        for y in range(grid_size):
            for x in range(grid_size):
                if counts_grid[y, x] > 0:
                    flow_grid[y, x] /= counts_grid[y, x]

        # Create visualization
        fig = go.Figure()

        # Add quiver plot for flow directions
        x_pos = np.linspace(cell_width / 2, img_width - cell_width / 2, grid_size)
        y_pos = np.linspace(cell_height / 2, img_height - cell_height / 2, grid_size)

        # Create meshgrid
        X, Y = np.meshgrid(x_pos, y_pos)

        # Get U and V components
        U = flow_grid[:, :, 0].flatten()
        V = flow_grid[:, :, 1].flatten()

        # Scale arrows based on magnitude
        scale = 0.02 * min(img_width, img_height)

        # Create heatmap for traffic density
        heatmap = counts_grid.copy()

        # Add heatmap
        fig.add_trace(go.Heatmap(
            z=heatmap,
            x=x_pos,
            y=y_pos,
            colorscale='Viridis',
            opacity=0.7,
            showscale=True,
            colorbar=dict(title='Count'),
            name='Traffic Density'
        ))

        # Add arrows for flow direction
        for i in range(grid_size):
            for j in range(grid_size):
                if counts_grid[i, j] > 0:
                    # Get position
                    x = x_pos[j]
                    y = y_pos[i]

                    # Get direction
                    dx = flow_grid[i, j, 0]
                    dy = flow_grid[i, j, 1]

                    # Add arrow
                    fig.add_trace(go.Scatter(
                        x=[x, x + dx * scale],
                        y=[y, y + dy * scale],
                        mode='lines',
                        line=dict(width=2, color='white'),
                        showlegend=False
                    ))

                    # Add arrowhead
                    angle = np.arctan2(dy, dx)
                    ax = x + dx * scale
                    ay = y + dy * scale

                    # Add arrowhead points
                    head_size = scale * 0.2
                    ah_1x = ax - head_size * np.cos(angle + np.pi / 6)
                    ah_1y = ay - head_size * np.sin(angle + np.pi / 6)
                    ah_2x = ax - head_size * np.cos(angle - np.pi / 6)
                    ah_2y = ay - head_size * np.sin(angle - np.pi / 6)

                    fig.add_trace(go.Scatter(
                        x=[ax, ah_1x, ah_2x, ax],
                        y=[ay, ah_1y, ah_2y, ay],
                        mode='lines',
                        fill='toself',
                        fillcolor='white',
                        line=dict(width=0, color='white'),
                        showlegend=False
                    ))

        # Update layout
        fig.update_layout(
            title="Traffic Flow Analysis",
            xaxis_title="X Position",
            yaxis_title="Y Position",
            yaxis=dict(autorange="reversed"),  # Reverse Y to match image coordinates
            height=800,
            width=800,
            showlegend=False
        )

        # Save if requested
        if output_path:
            try:
                # Change file extension to html if it's not already
                html_path = output_path.replace('.png', '.html')
                fig.write_html(html_path)
                print(f"Flow analysis saved to {html_path}")
            except Exception as e:
                warnings.warn(f"Could not save flow analysis to file: {e}")
                print(f"Warning: Could not save flow analysis to file: {e}")
                print("Continuing with analysis...")

        return flow_grid, fig

    def analyze_and_save_all(self, output_dir):
        """Run all analyses and save results.

        Args:
            output_dir: Directory to save results
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Trajectory metrics
        print("Analyzing trajectories...")
        metrics = self.analyze_trajectories()

        try:
            metrics.to_csv(os.path.join(output_dir, 'trajectory_metrics.csv'), index=False)
            print(f"Saved trajectory metrics to {os.path.join(output_dir, 'trajectory_metrics.csv')}")
        except Exception as e:
            warnings.warn(f"Could not save trajectory metrics to file: {e}")
            print(f"Warning: Could not save trajectory metrics to file: {e}")

        # Trajectory map
        print("Generating trajectory map...")
        fig = self.plot_trajectory_map(
            min_frames=5,
            output_path=os.path.join(output_dir, 'trajectory_map.html')  # Using HTML instead of PNG
        )

        # Speed distributions
        print("Analyzing speed distributions...")
        self.plot_speed_distributions(
            output_path=os.path.join(output_dir, 'speed_distributions.png')
        )

        # Flow analysis
        print("Performing flow analysis...")
        flow_grid, flow_fig = self.flow_analysis(
            grid_size=8,
            output_path=os.path.join(output_dir, 'flow_analysis.html')  # Using HTML instead of PNG
        )

        print(f"Analysis complete. Results saved to {output_dir}")
        print(f"Interactive visualizations saved as HTML files. Open them in a web browser to explore.")


# Example usage
if __name__ == "__main__":
    # Load the tracking data from the previous step
    tracking_csv = "../results/tracked_video_tracking.csv"

    # Create analyzer
    analyzer = TrajectoryAnalyzer(tracking_csv)

    # Run analyses and save results
    analyzer.analyze_and_save_all("../results/trajectory_analysis")