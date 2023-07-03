FROM python:3.9

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg
RUN apt-get update && apt-get install -y ffprobe

# Install python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
