# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.8
WORKDIR /pipelines
COPY requirements.txt /pipelines
RUN pip install -r requirements.txt
COPY inference_service.py /pipelines

RUN chmod +x /pipelines/inference_service.py

ENTRYPOINT ["python"]
CMD ["/pipelines/inference_service.py"]
