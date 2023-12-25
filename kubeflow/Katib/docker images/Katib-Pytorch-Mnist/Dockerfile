FROM python:3.9-slim



WORKDIR /usr/app/src

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY mnist.py ./

CMD ["python", "./mnist.py"]