from keras.models import load_model
import argparse
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from PIL import Image

# Predicts the class for a single input image.
def predict(args):
    #Initialize the model.
    model = load_model('model.h5')

    # Get image path from user argument
    image_path = args.input
    # Create a numpy representation of each image
    image = np.array(Image.open(image_path))
    # Resize for uniformity and scale between 0 and 1
    image = np.resize(image,(IMG_SIZE,IMG_SIZE,3)) / 255
    # Expand dimensions so the model accepts one input
    image = np.expand_dims(image, 0)

    #Predict the class of the image - returned as single item list of floats between 0 and 1
    prediction = model.predict(image)
    
    # Return the first and only item in list of predicitons
    return prediction[0]


if __name__ == "__main__":
    # Global varibles
    IMG_SIZE = 250
    
    # Class labels for intepreting classifier output
    classes = ['Outdoor' , 'Indoor']
    
    # Set up an argument parser and retrieve the image filename for classification
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='filename for the image being classified')
    args = parser.parse_args()
    
    # Round returned float to 0 or 1 to get prediction class index
    prediction = np.round(predict(args))
     
    # Display filename and model prediction
    print("File name: " + args.input)
    print("Prediction: " + classes[prediction[0].astype(int)])
