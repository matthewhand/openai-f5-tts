from setuptools import setup, find_packages
import os

setup(
    name="openai_f5_tts",
    version="0.1.0",
    packages=find_packages(exclude=["tests*", "docs"]),
    install_requires=[
        "flask",
        "gevent",
        "python-dotenv",
        "transformers>=4.46.2,<5.0.0",
        "tokenizers>=0.20.3,<1.0.0",
        "torch>=2.0.0",
        "librosa>=0.9.2",
        "soundfile>=0.11.0",
        "numpy>=1.23.0",
        "safetensors>=0.2.5",
    ],
    entry_points={
        'console_scripts': [
            'openai-f5-tts=app.cli:main',
        ],
        'tts_engines': [
            'f5 = app.engines.f5:Engine',
            'kokoro = app.engines.kokoro:Engine',
        ],
    },
    author="MatthewH",
    description="Flask-based API for F5-TTS with expression parsing and CLI",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/matthewhand/openai-f5-tts",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
