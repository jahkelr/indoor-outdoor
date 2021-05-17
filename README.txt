File Descriptions:
 - vocabulary.csv: provided file containing information for corresponding label data information
 - model.h5: file containing the model architecture and weights of the pre-trained classifier
 - video_category_data.json: provided file containing label information for the videos associated with the thumbnail files
 - classify.py: file used to load classifier and and input image. Outputs a classification label
 - train_classifier.py: file used to train classifier after loading the dataset. Provides evaluation information once model is trained
 - unit_tests.py: file used to run a unit test for the create model function

Directories:
 - images/: contains the provided images from the YT-8M dataset, split into Indoor and Outdoor directories
 - Evaluation Output/: Images of the model accuracy and loss graphs and training time information 

How to use the classifier:

 Classifier can be called by using the following command: python3 classify.py --input <FILE_PATH>

 - --input is a required argument that takes an image file path as its only argument

How to run unit test:

 Unit test can be run by using the following command: python3 unit_tests.py
 
 Enviornment should be equipped with python3 and tensorflow
 pip3 install --user --upgrade tensorflow  # install in $HOME
