import unittest
import tensorflow as tf
import train_classifier as tc
from tensorflow.keras.models import Sequential

class UnitTest_Model_create(unittest.TestCase):
    
    # This is a test to verify the model we create for training is a sequential model
    def test_upper(self):
        # Call the imported function
        model = tc.create_model()
        # Create model to extract type
        seq_test = Sequential()
        
        self.assertTrue(isinstance(model, type(seq_test)))

if __name__ == '__main__':
    unittest.main()
