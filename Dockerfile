FROM python:3.11.4
# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg ffprobe

# Install python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
