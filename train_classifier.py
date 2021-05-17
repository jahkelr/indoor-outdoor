import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

"""
This function handles files.

Convert the the images to numpy arrays for model training.

@params:
    - image_size: Height and width for each image resize
"""
def get_image_data(image_size):
    image_data = [] # Array of np.array(img)
    label_data = [] # Array of manual labels - 0: Outdoor, 1: Indoor
    
    images_folder = './images/indoor/'
    #Create dataset for indoor images
    for file in os.listdir(images_folder):
        image_path = images_folder + file
        #Create a numpy representation of each image
        image = np.array(Image.open(image_path))
        #Resize for uniformity and scale between 0 and 1
        image = np.resize(image,(image_size,image_size,3)) / 255
        #Add to array for image dataset
        image_data.append(image)
        label_data.append(1) #Class: Indoor
    
    images_folder = './images/outdoor/'
    
    for file in os.listdir(images_folder):
        image_path = images_folder + file
        #Create a numpy representation of each image
        image = np.array(Image.open(image_path))
        #Resize for uniformity and scale between 0 and 1
        image = np.resize(image,(IMG_SIZE,IMG_SIZE,3)) / 255
        #Add to array for image dataset
        image_data.append(image)
        label_data.append(0) #Class: Outdoor

    return image_data, label_data

"""
Create a CNN with a single classificiation node for Indoor vs Outdoor
@params:
    img_size: Height and width for each image resize
    dropout: Value for dropout layer(s) within model
"""
def create_model(img_size=200, dropout=0.2):
    model = Sequential()
    # Four layers of Convulation for feature extraction and max pooling to reduce dimensionality
    model.add(Conv2D(32, (3, 3), input_shape=(img_size, img_size, 3) , activation="relu")) # Input layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation="relu"))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3) , activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Fully connected layer to detect final high level features.
    model.add(Flatten()) # Flatten input for Dense layers
    model.add(Dropout(dropout))
    
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    
    model.add(Dropout(dropout)) # Dropout
    model.add(Dense(1, activation="sigmoid")) #Classifier node
    model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam")
    
    # Show us what we made
    model.summary()
    
    return model

"""
This function builds a CNN and fits the model on training
data that is shuffled in a reproducable "random" state.
Returns model and history.
"""
def train_CNN_on_dataset():
    # Call function to build dataset given an image size
    image_data , label_data = get_image_data(IMG_SIZE)
    
    # Change to numpy array representations for split and training
    image_data , label_data = np.array(image_data) , np.array(label_data)
            
    # Split into train and validation sets
    X_train , X_test, y_train, y_test = train_test_split( image_data, label_data, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        
    # Build the image classifier
    model = create_model(IMG_SIZE,DROPOUT)
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=EPOCHS,validation_data=(X_test, y_test))
    
    return history, model
    
if __name__ == "__main__":
    # Global varibles
    IMG_SIZE = 250
    DROPOUT = 0.4
    RANDOM_STATE = 82
    EPOCHS = 20
    TEST_SIZE = 0.2

    # train a classifier based on parameters
    history, model = train_CNN_on_dataset()
    
    #Save the model for classification
    model.save("model.h5")
    
    # Show us how the training went
    # plots keras
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
