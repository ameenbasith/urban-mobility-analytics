from setuptools import setup, find_packages

setup(
    name="urban-mobility-analytics",
    version="0.1",
    packages=find_packages(),
    python_requires=">=3.9,<3.12",
    install_requires=[
        "streamlit>=1.24.0",
        "opencv-python-headless>=4.7.0",
        "numpy>=1.23.0",
        "pandas>=1.5.0",
        "plotly>=5.13.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "imageio>=2.30.0",
        "Pillow>=10.0.0",
    ],
    extras_require={
        "models": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "ultralytics>=8.0.20"
        ]
    }
)