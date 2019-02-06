# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 00:39:31 2019

@author: miljan
"""
import numpy as np
import cv2
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

def scale_to_range(image): # skalira elemente slike na opseg od 0 do 1
    ''' Elementi matrice image su vrednosti 0 ili 255. 
        Potrebno je skalirati sve elemente matrica na opseg od 0 do 1
    '''
    return image/255
def matrix_to_vector(image):
    '''Sliku koja je zapravo matrica 28x28 transformisati u vektor sa 784 elementa'''
    return image.flatten()
def prepare_for_ann(regions):
    '''Regioni su matrice dimenzija 28x28 čiji su elementi vrednosti 0 ili 255.
        Potrebno je skalirati elemente regiona na [0,1] i transformisati ga u vektor od 784 elementa '''
    ready_for_ann = []
    for region in regions:
        # skalirati elemente regiona 
        # region sa skaliranim elementima pretvoriti u vektor
        # vektor dodati u listu spremnih regiona
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))
def prepare_outputs_for_ann(outputs):
    #za svaki podataka iz y_train kreiram niz od 10 elementa sa 1 na odgovarajucem mjestu
    ann_outputs = []
    for number in outputs:
        output = np.zeros(10)
        output[number] = 1
        ann_outputs.append(output)
    return np.array(ann_outputs)

def create_ann():
    '''Implementacija veštačke neuronske mreže sa 784 neurona na uloznom sloju,
        128 neurona u skrivenom sloju i 10 neurona na izlazu. Aktivaciona funkcija je sigmoid.
    '''
    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(10, activation='sigmoid'))
    return ann
    
def train_ann(ann, X_train, y_train):
    '''Obucavanje vestacke neuronske mreze'''
    X_train = np.array(X_train, np.float32) # dati ulazi
    y_train = np.array(y_train, np.float32) # zeljeni izlazi za date ulaze  
    # definisanje parametra algoritma za obucavanje
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)
    # obucavanje neuronske mreze
    ann.fit(X_train, y_train, epochs=500, batch_size=1, verbose = 0, shuffle=False) 
      
    return ann

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train_binary = []

for img in x_train[:5000:]:
   ret, binary = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
   x_train_binary.append(binary)
    
ann = create_ann()
ann = train_ann(ann,np.array(prepare_for_ann(x_train),np.float32),np.array(prepare_outputs_for_ann(y_train),np.float32))
model_json = ann.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
ann.save_weights("model.h5")
print("Saved model to disk")
