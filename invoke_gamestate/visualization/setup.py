from setuptools import setup, find_packages

setup(
    name="visualization",
    version="0.0.1",
    description="Soccer game state visualization tools",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "pydantic",
        "tqdm",
    ],
    python_requires=">=3.7",
)
