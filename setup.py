"""
GDA Smart Building – setup.py
"""
from setuptools import setup, find_packages

setup(
    name="gda_smart_building",
    version="1.0.0",
    description=(
        "Scalable Activity Recognition in Smart Buildings via "
        "Generalized Domain Adaptation of IoT Sensor Data"
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Smart Building GDA Research",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    install_requires=[
        "torch>=2.0",
        "numpy>=1.24",
        "scikit-learn>=1.2",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "tqdm>=4.65",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "jupyter", "ipykernel"],
        "viz": ["plotly>=5.0", "pandas>=2.0"],
    },
    entry_points={
        "console_scripts": [
            "gda-quickstart=scripts.quickstart:main",
            "gda-benchmark=scripts.run_benchmark:main",
            "gda-generate=scripts.generate_data:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
)
