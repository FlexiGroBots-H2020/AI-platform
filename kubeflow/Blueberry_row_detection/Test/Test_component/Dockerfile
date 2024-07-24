# Use the official GDAL base image
FROM ghcr.io/osgeo/gdal:ubuntu-full-latest

# Set the working directory to /app
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

# Install additional Python libraries and dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    libopencv-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with GPU support
RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision -f https://download.pytorch.org/whl/cu111/torch_stable.html

# Install other libraries
RUN pip3 install --upgrade numpy
RUN pip3 install opencv-python-headless
RUN pip3 install segmentation-models-pytorch
RUN pip3 install matplotlib

# Copy the current directory contents into the container at /app
COPY . /app

# Run delineation_and_zones_main.py when the container launches
CMD ["python3", "test_pipeline.py"]
