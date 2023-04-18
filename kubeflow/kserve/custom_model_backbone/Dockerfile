FROM python:3.8-slim

WORKDIR /pipelines

COPY requirements.txt /pipelines
RUN pip install -r requirements.txt

COPY inference_service_kserve.py /pipelines

ENTRYPOINT ["python","/pipelines/inference_service_kserve.py"]