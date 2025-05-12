import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os


class TrafficAnalyzer:
    def __init__(self, detection_data):
        """Initialize with detection data DataFrame or CSV path.

        Args:
            detection_data: DataFrame or path to CSV with detection data
        """
        if isinstance(detection_data, str):
            self.data = pd.read_csv(detection_data)
        else:
            self.data = detection_data.copy()

        # Add derived columns if they don't exist
        if 'timestamp' not in self.data.columns:
            self.data['timestamp'] = self.data['frame'] / self.data['fps']

        print(f"Loaded detection data with {len(self.data)} objects")

    def count_by_class(self):
        """Count objects by class."""
        class_counts = self.data['class_name'].value_counts()
        return class_counts

    def count_over_time(self, time_window=5):
        """Count objects over time windows.

        Args:
            time_window: Time window in seconds

        Returns:
            DataFrame with counts by time window and class
        """
        # Create time bins
        max_time = self.data['timestamp'].max()
        bins = np.arange(0, max_time + time_window, time_window)

        # Add time window column
        self.data['time_window'] = pd.cut(self.data['timestamp'], bins)

        # Group by time window and class
        counts = self.data.groupby(['time_window', 'class_name']).size().unstack(fill_value=0)

        # Add total column
        counts['total'] = counts.sum(axis=1)

        return counts

    def plot_counts_by_class(self, output_path=None):
        """Plot counts by class.

        Args:
            output_path: Path to save the plot (optional)
        """
        counts = self.count_by_class()

        plt.figure(figsize=(12, 6))
        sns.barplot(x=counts.index, y=counts.values)
        plt.title('Object Counts by Class')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path)
            print(f"Plot saved to {output_path}")

        plt.show()

    def plot_counts_over_time(self, output_path=None):
        """Plot counts over time.

        Args:
            output_path: Path to save the plot (optional)
        """
        counts = self.count_over_time()

        plt.figure(figsize=(15, 8))

        # Drop total for this plot
        plot_data = counts.drop('total', axis=1) if 'total' in counts.columns else counts

        # Plot each class
        plot_data.plot(kind='line', marker='o', ax=plt.gca())

        plt.title('Object Counts Over Time')
        plt.xlabel('Time Window')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.legend(title='Class')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path)
            print(f"Plot saved to {output_path}")

        plt.show()

    def generate_heatmap(self, image_width, image_height, grid_size=32, output_path=None):
        """Generate a heatmap of object locations.

        Args:
            image_width: Width of the original video
            image_height: Height of the original video
            grid_size: Size of grid cells
            output_path: Path to save the heatmap (optional)

        Returns:
            Numpy array with heatmap data
        """
        # Create grid
        x_bins = np.linspace(0, image_width, grid_size + 1)
        y_bins = np.linspace(0, image_height, grid_size + 1)

        # Create bin labels
        x_labels = [f"{x:.0f}" for x in x_bins[:-1]]
        y_labels = [f"{y:.0f}" for y in y_bins[:-1]]

        # Create 2D histogram using object centers
        heatmap, _, _ = np.histogram2d(
            self.data['center_x'],
            self.data['center_y'],
            bins=[x_bins, y_bins]
        )

        # Plot heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(heatmap.T, cmap='viridis')
        plt.title('Object Location Heatmap')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path)
            print(f"Heatmap saved to {output_path}")

        plt.show()

        return heatmap

    def analyze_and_save_all(self, image_width, image_height, output_dir):
        """Run all analyses and save results.

        Args:
            image_width: Width of the original video
            image_height: Height of the original video
            output_dir: Directory to save results
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Counts by class
        counts = self.count_by_class()
        counts.to_csv(os.path.join(output_dir, 'class_counts.csv'))
        self.plot_counts_by_class(os.path.join(output_dir, 'class_counts.png'))

        # Counts over time
        time_counts = self.count_over_time()
        time_counts.to_csv(os.path.join(output_dir, 'time_counts.csv'))
        self.plot_counts_over_time(os.path.join(output_dir, 'time_counts.png'))

        # Heatmap
        heatmap = self.generate_heatmap(
            image_width, image_height,
            output_path=os.path.join(output_dir, 'heatmap.png')
        )
        np.save(os.path.join(output_dir, 'heatmap.npy'), heatmap)

        print(f"All analyses saved to {output_dir}")


# Example usage
if __name__ == "__main__":
    # Load the detection data from the previous step
    detection_csv = "../results/processed_video_detections.csv"

    # Create analyzer
    analyzer = TrafficAnalyzer(detection_csv)

    # Run analyses and save results
    analyzer.analyze_and_save_all(1920, 1080, "../results/analysis")