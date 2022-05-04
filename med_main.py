# Main code for the project - Preprocessing

import numpy as np
import os
import cv2
import pickle
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit

import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.models import Sequential

DATADIR = "..\\PAap\\Flowers299\\Flowers299"
CATEGORIES = ['Jasmine', 'JohnnyJumpUp', 'KaffirLily', 'Lantana', 'Larkspur', 'LoveintheMist', 'Lunaria', 'TeaRose', 'TigerFlower', 'TrumpetVine']
dataset = []
IMG_SIZE = 224

def create_dataset():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)

        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                dataset.append([new_array, class_num])
            except Exception as e:
                pass


create_dataset()

X = []
y = []
for features, label in dataset:
    X.append(features)
    y.append(label)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y = np.array(y)
X = X/255.0
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8, random_state=0)
sss.get_n_splits(X, y)

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# pickle_out = open("X_train.pickle", "wb")
# pickle.dump(X, pickle_out)
# pickle_out.close()
#
# pickle_out = open("y_train.pickle", "wb")
# pickle.dump(X, pickle_out)
# pickle_out.close()
#
# pickle_out = open("X_test.pickle", "wb")
# pickle.dump(X, pickle_out)
# pickle_out.close()
#
# pickle_out = open("y_test.pickle", "wb")
# pickle.dump(y, pickle_out)
# pickle_out.close()
class_num = len(CATEGORIES)

model = Sequential([
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(class_num)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs = 7
history = model.fit(
  X_train, y_train,
  validation_data=(X_test, y_test),
  epochs=epochs
)
history


