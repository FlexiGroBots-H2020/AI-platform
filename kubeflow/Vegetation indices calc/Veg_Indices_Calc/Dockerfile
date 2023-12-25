# Use the official GDAL base image
FROM ghcr.io/osgeo/gdal:ubuntu-full-latest

# Set the working directory to /app
WORKDIR /app

# Install additional Python libraries
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
RUN pip3 install cython
RUN pip3 install scikit-learn
RUN pip3 install scikit-image
RUN pip3 install numpy
RUN pip3 install opencv-python-headless

# Copy the current directory contents into the container at /app
COPY . /app

# Compile Cython modules
RUN python3 setup_cython_function.py build_ext --inplace

# Run veg_indices_calc_pipeline.py when the container launches
CMD ["python3", "veg_indices_calc_pipeline.py"]
