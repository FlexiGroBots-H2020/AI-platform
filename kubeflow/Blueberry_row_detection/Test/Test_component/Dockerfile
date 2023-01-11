FROM continuumio/miniconda3
WORKDIR /pipelines
COPY requirements.txt /pipelines

# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]

COPY environment.yml .

RUN pip install -r requirements.txt



RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y


RUN conda env create -f environment.yml

# Initialize conda in bash config fiiles:
RUN conda init bash

RUN conda activate myenv

RUN conda install gdal

RUN conda install pytorch torchvision torchaudio cudatoolkit matplotlib tensorboard -c pytorch



COPY . /pipelines

RUN chmod +x /pipelines/test_pipeline.py

ENTRYPOINT ["python"]
CMD ["/pipelines/test_pipeline.py"]
