#!/bin/bash
set -e

sudo apt update
sudo apt install -y \
  python3 python3-venv python3-pip \
  espeak ffmpeg libsndfile1 portaudio19-dev \
  libcairo2-dev libgirepository1.0-dev \
  gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0

# espeak (eSpeak) is required for the offline TTS flow used by emo_v4.py.
# libsndfile1 and portaudio are required for soundfile and sounddevice (used when playing WAVs).
# ffmpeg is optional but useful if you need to convert audio formats or debug audio files.

# Set PYGLFW_LIBRARY_VARIANT=x11 permanently (fixes MuJoCo GUI on Wayland)
if ! grep -q "PYGLFW_LIBRARY_VARIANT" ~/.bashrc; then
  echo 'export PYGLFW_LIBRARY_VARIANT=x11' >> ~/.bashrc
  echo "Added PYGLFW_LIBRARY_VARIANT=x11 to ~/.bashrc"
fi
