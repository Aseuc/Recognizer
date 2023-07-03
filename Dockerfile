FROM python:3.10
# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg

# Install python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
