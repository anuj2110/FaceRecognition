# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 23:09:25 2020

@author: Anuj
"""
import os
from shutil import copyfile
import glob
import random

base_dir = "./Images/"
train_dir = "./Images/train/"
test_dir = "./Images/test/"

os.mkdir(train_dir)
os.mkdir(test_dir)

person_names = os.listdir(base_dir)[:-2]

train_dirs = []
test_dirs =[]

for person_name in person_names:
    train_dirs.append(train_dir+person_name+"/")
    test_dirs.append(test_dir+person_name+"/")
    os.mkdir(train_dir+person_name+"/")
    os.mkdir(test_dir+person_name+"/")

base_dirs =[]
for person_name in person_names:
    base_dirs.append(base_dir+person_name+"/")
    

def make_train_test_set(source_dir,train_dir,test_dir,split_size=0.9):
    files=[]
    for filename in os.listdir(source_dir):
        file = source_dir+filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")
            
    training_length = int(len(files) * split_size)
    testing_length = int(len(files) - training_length)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    testing_set = shuffled_set[-testing_length:]
    
    for filename in training_set:
        this_file = source_dir + filename
        destination = train_dir + filename
        copyfile(this_file, destination)

    for filename in testing_set:
        this_file = source_dir + filename
        destination = test_dir + filename
        copyfile(this_file, destination)

for base_dir,train_dir,test_dir in zip(base_dirs,train_dirs,test_dirs):
    
    make_train_test_set(base_dir,train_dir,test_dir)
