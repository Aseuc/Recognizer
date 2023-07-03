FROM python:3.10
# Install system dependencies
RUN apt-get update && apt-get install ffmpeg
RUN apt-get update && apt-get install ffprobe
# Install python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN /home/appuser/venv/bin/python -m pip install --upgrade pip
