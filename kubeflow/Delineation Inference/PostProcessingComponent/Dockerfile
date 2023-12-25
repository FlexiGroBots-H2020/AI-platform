# Start from a base image with Python 3.8
FROM python:3.9-slim

# Set the working directory to /app
WORKDIR /app

# Install the Minio Python library
RUN pip install minio

# Copy the current directory contents into the container at /app
COPY . /app

# Run the script when the container launches
CMD ["python", "-u", "your_script.py"]
