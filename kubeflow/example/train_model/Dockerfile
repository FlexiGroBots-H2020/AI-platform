# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.8
WORKDIR /pipelines
COPY requirements.txt /pipelines
RUN pip install -r requirements.txt
COPY train_model.py /pipelines

RUN chmod +x /pipelines/train_model.py

ENTRYPOINT ["python"]
CMD ["/pipelines/train_model.py"]
