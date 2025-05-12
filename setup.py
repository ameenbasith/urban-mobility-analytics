from setuptools import setup, find_packages

setup(
    name="urban-mobility-analytics",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "opencv-python-headless",
        "numpy",
        "pandas",
        "plotly",
        "matplotlib",
        "seaborn",
        "torch",
        "torchvision",
        "imageio",
        "Pillow",
    ],
)