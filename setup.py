from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="talknet_asd",
    version="0.1.0",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.6.0",
        "torchaudio>=0.6.0",
        "numpy",
        "scipy",
        "scikit-learn",
        "tqdm",
        "scenedetect",
        "opencv-python",
        "python_speech_features",
        "torchvision",
        "ffmpeg",
        "gdown",
        "youtube-dl",
        "ultralytics",
        "PyTurboJPEG",
    ],
) 