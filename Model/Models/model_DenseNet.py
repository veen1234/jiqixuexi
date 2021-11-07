# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 10:43:07 2020

@author: Study
"""
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Add, BatchNormalization, LeakyReLU
from keras.layers import Dropout, concatenate, Input,  GlobalAveragePooling2D, Activation, Reshape, multiply
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import plot_model
from keras import layers


def densenet(input_size, num_class, pretrained_weights = False, model_summary=False, model_plot=False):

    inputs = Input(input_size)
    
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same')(inputs)
    
    cv1 = DenseBlock(conv1, 32)
    
    cv1 = BatchNormalization(axis=3)(cv1)
    cv1 = LeakyReLU(alpha=0.3)(cv1)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'valid')(cv1)
    
    cv2 = DenseBlock(conv2, 64)
    
    cv2 = BatchNormalization(axis=3)(cv2)
    cv2 = LeakyReLU(alpha=0.3)(cv2)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'valid')(cv2)
    
    cv3 = DenseBlock(conv3, 128)
    
    pool1 = GlobalAveragePooling2D()(cv3)

    fc1 = Dense(units=num_class, activation="relu")(pool1)
    fc2 = Dense(units = num_class,activation="softmax")(fc1) #fc_size
    
    model = Model(inputs = inputs, outputs = fc2)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    #model.compile(optimizer = SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    if model_summary is True:
        model.summary()

    if pretrained_weights is True:
        model.load_weights(pretrained_weights)

    if model_plot is True:
        plot_model(model, to_file='net.png')

    return model

def DenseBlock(x, cv_size):
    
    # x1 = SEnet(x1, cv_size)
    conv1 = Conv2D(cv_size, 3, activation = 'relu', padding = 'same')(x)
    
    y1 = concatenate([x, conv1], axis = 3)
    x1 = SEnet(y1, (2*cv_size))
    conv2 = Conv2D(cv_size, 3, activation = 'relu', padding = 'same')(x1)
    
    y2 = concatenate([x, conv1, conv2], axis = 3)
    x2 = SEnet(y2, (3*cv_size))
    conv3 = Conv2D(cv_size, 3, activation = 'relu', padding = 'same')(x2)
    
    y3 = concatenate([x, conv1, conv2, conv3], axis = 3)
    x3 = SEnet(y3, (4*cv_size))
    conv4 = Conv2D(cv_size, 3, activation = 'relu', padding = 'same')(x3)
    
    return conv4

def DenseBlock1(x, cv_size):
    
    conv1 = Conv2D(cv_size, 3, activation = 'relu', padding = 'same')(x)
    x1 = SEnet(x, cv_size)
    
    y1 = concatenate([x1, conv1], axis = 3)
    conv2 = Conv2D(cv_size, 3, activation = 'relu', padding = 'same')(y1)
    x2 = SEnet(conv2, cv_size)
    
    y2 = concatenate([x1, x2, conv2], axis = 3)
    conv3 = Conv2D(cv_size, 3, activation = 'relu', padding = 'same')(y2)
    x3 = SEnet(conv3, cv_size)
    
    y3 = concatenate([x1, x1, x3, conv3], axis = 3)
    conv4 = Conv2D(cv_size, 3, activation = 'relu', padding = 'same')(y3)
    
    return conv4

def SEnet(x, fc_size):
    # Squeeze And Excitation
    squeeze = GlobalAveragePooling2D()(x)
    excitation = Dense(units=fc_size // 16)(squeeze)
    excitation = Activation('relu')(excitation)
    excitation = Dense(units=fc_size)(excitation)
    excitation = Activation('sigmoid')(squeeze)
    excitation = Reshape((1, 1, fc_size))(excitation)
    y = multiply([x, excitation])
    
    return y
    
if __name__=='__main__':
    
    densenet(input_size=(25,25,32), num_class=9, model_summary=True)


