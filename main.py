import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from sklearn.metrics import classification_report, log_loss, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

data_dir = '../PAap/Flowers299/Flowers299'

Name=[]
for file in os.listdir(data_dir):
    if re.search(r'ÔÇÖ', file):
        file = file.replace("ÔÇÖ", "'")
    Name+=[file]

print(Name)
print(len(Name))

N=[]
for i in range(len(Name)):
    N+=[i]

normal_mapping = dict(zip(Name, N))
reverse_mapping = dict(zip(N, Name))

def mapper(value):
    return reverse_mapping[value]

dataset = []
count = 0
for file in tqdm(os.listdir(data_dir)):
    path=os.path.join(data_dir, file)
    for im in os.listdir(path):
        image = load_img(os.path.join(path,im), grayscale = False, color_mode='rgb', target_size=(40,40))
        image = img_to_array(image)
        image = image/255.0
        dataset += [[image,count]]
    count=count+1


n=len(dataset)
print(n)

num=[]
for i in range(n):
    num+=[i]
random.shuffle(num)
print(num[0:51])

data, labels=zip(*dataset)
data=np.array(data)
labels=np.array(labels)
