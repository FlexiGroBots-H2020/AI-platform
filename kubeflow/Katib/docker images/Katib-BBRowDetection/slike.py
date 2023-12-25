import torch
import os
from PIL import Image
import numpy as np

for pic in os.listdir("Slike/"):
    img = Image.open("Slike/" + pic)
    nparray = np.array(img)
    print(nparray)
    break