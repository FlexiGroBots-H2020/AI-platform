FROM python:3.8
WORKDIR /pipelines
COPY requirements.txt /pipelines

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y


RUN pip install -r requirements.txt
COPY . /pipelines

RUN chmod +x /pipelines/Train_BGFG_BCE_with_weightsUnet3.py

ENTRYPOINT ["python"]
CMD ["/pipelines/Train_BGFG_BCE_with_weightsUnet3.py"]
