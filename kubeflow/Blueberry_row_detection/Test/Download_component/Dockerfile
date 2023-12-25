FROM python:3.8
WORKDIR /pipelines
COPY requirements.txt /pipelines

RUN pip install -r requirements.txt
COPY . /pipelines

RUN chmod +x /pipelines/download_data.py

ENTRYPOINT ["python"]
CMD ["/pipelines/download_data.py"]
