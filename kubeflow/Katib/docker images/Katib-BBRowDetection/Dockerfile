FROM python:3.9-slim



WORKDIR /usr/app/src

COPY FullSet .

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY Train_BGFG_BCE_with_weightsUnet3.py ./

COPY Unet_LtS.py ./

COPY model_utils.py ./

COPY metrics_utils.py ./

COPY loss_utils.py ./

COPY data_utils.py ./

COPY configUnet3.py ./

COPY SegNet.py ./

COPY PSPNet.py ./

COPY UperNet.py ./

COPY DUC_HDCNet.py ./

CMD ["python", "./Train_BGFG_BCE_with_weightsUnet3.py"]
