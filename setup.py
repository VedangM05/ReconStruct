"""Setup script for ReconStruct package."""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="reconstruct",
    version="0.1.0",
    description="2D Blueprint to 3D Model Conversion System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ReconStruct",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "scikit-image>=0.21.0",
        "trimesh>=3.20.0",
        "pyvista>=0.42.0",
        "networkx>=3.1",
        "shapely>=2.0.0",
        "Flask>=2.3.0",
        "flask-cors>=4.0.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "pillow>=10.0.0",
        "tqdm>=4.66.0",
        "jsonschema>=4.19.0",
    ],
    extras_require={
        "ml": ["tensorflow>=2.13.0", "torch>=2.0.0", "torchvision>=0.15.0"],
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0", "black>=23.0.0", "flake8>=6.0.0", "mypy>=1.5.0"],
    },
    entry_points={
        "console_scripts": [
            "reconstruct=src.main:main",
        ],
    },
    include_package_data=True,
)
