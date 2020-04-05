# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 13:21:54 2020

@author: Anuj
"""

from tensorflow.keras import layers as l
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.applications.vgg16 import VGG16
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from glob import glob
from tensorflow.keras.preprocessing import image

train_path = "./Images/train"
test_path = "./Images/test"

folders = glob("./Images/train/*")

labels = len(folders)


