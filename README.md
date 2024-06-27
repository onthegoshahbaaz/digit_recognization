# digit_recognization
A system capable of accurately identifying handwritten digits. 
MNIST Convolutional Neural Network (CNN) with TensorFlow/Keras
This repository contains a script to train a Convolutional Neural Network (CNN) on the MNIST dataset using TensorFlow and Keras. The MNIST dataset consists of handwritten digit images, where each image is a grayscale image of size 28x28 pixels.

Dependencies
Ensure you have the following dependencies installed:

Python 3.x
TensorFlow 2.x
NumPy
Matplotlib
scikit-learn (for metrics)
You can install TensorFlow and other Python dependencies using pip:

pip install tensorflow numpy matplotlib scikit-learn
Dataset
The MNIST dataset is used for training and testing the CNN.

Training images (train-images-idx3-ubyte): 60,000 images
Training labels (train-labels-idx1-ubyte): 60,000 labels
Test images (test-images-idx3-ubyte): 10,000 images
Test labels (test-labels-idx1-ubyte): 10,000 labels


Execute the Python script mnist_cnn.py to train and evaluate the CNN model:
This will train the model on the training data, evaluate its performance on the test data, and print the test accuracy and confusion matrix.

Model Architecture
The CNN model architecture used for training:

Input Layer: Conv2D with 32 filters, each 3x3, ReLU activation
MaxPooling Layer: 2x2 pooling
Convolutional Layer: Conv2D with 64 filters, each 3x3, ReLU activation
MaxPooling Layer: 2x2 pooling
Flatten Layer: Flattens the input
Dense Layer: Fully connected layer with 64 units, ReLU activation
Output Layer: Dense layer with 10 units (one for each digit), softmax activation
Results
After training for 5 epochs, the model achieves a test accuracy of approximately 98%. The confusion matrix provides insights into the model's performance across different digit classes.

Further Improvements
Experiment with different CNN architectures (e.g., adding more layers, adjusting filter sizes).
Fine-tune hyperparameters such as learning rate, batch size, and number of epochs.
Explore data augmentation techniques to improve generalization.

