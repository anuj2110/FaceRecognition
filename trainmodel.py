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
import matplotlib.pyplot as plt

train_path = "./Images/train"
test_path = "./Images/test"

folders = os.listdir(train_path)

labels = len(folders)

target_size = [224,224]

train_generator = ImageDataGenerator(rescale=1/255,
                                     horizontal_flip = True,
                                     shear_range = 0.2,
                                     zoom_range = 0.2)

test_generator = ImageDataGenerator(rescale=1/255)


vgg = VGG16(input_shape=target_size+[3], weights = 'imagenet', include_top=False)

for layer in vgg.layers:
    layer.trainable = False
    

x = l.Flatten()(vgg.output)

prediction = l.Dense(labels,activation = 'softmax')(x)

model = Model(inputs = vgg.input,outputs = prediction)
model.summary()

# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

training_set = train_generator.flow_from_directory(train_path,
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_generator.flow_from_directory(test_path,
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')



r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=5,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)
# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.savefig('LossVal_loss.png')
plt.show()


# accuracies
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.savefig('AccVal_acc.png')
plt.show()


model.save('facefeatures_new_model.h5')