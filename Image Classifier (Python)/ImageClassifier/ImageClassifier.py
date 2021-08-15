'''
Keras is a high-level neural network library that runs on top of TensorFlow
https://keras.io/api/datasets/fashion_mnist/ (output layer would be between 0 to 9, )
Means input layer has 60000 images, then their is one hidden layer, lastly the output layer has 0-9 images
Means input layer has 784 nodes, then their is one hidden layer with 128 nodes, lastly the output layer has 10 nodes

In this project, I am just playing with image matrix and in real life I can play with stocks, some trends, etc. and they involve matrices and palying it
'''

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt  # print stuff

# load pre-defined dataset, has one image, 28*28, black means 0 and white means some number
fashion_mnist = keras.datasets.fashion_mnist

# 60000 images for training and 10000 for testing
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  # pull out data from dataset

# show data
print(train_labels[0])
print(train_images[0])
# train_images[1] => b/w 0 to 60000, but e.g. train_images[1] means I am focusing to train the shirt
plt.imshow(train_images[100], cmap='gray', vmin=0, vmax=255)
plt.show()

'''
0-255 means 256, 2^8 means 8 bits.
Neural network has nodes and each node has 1 pixel
Input layer = hidden layers (capture some patterns) = output layer
sequential means divide vertically and then they will form a column 
'''

# Define our neural net structure
model = keras.Sequential([
    # Input layer is 28*28 image matrix and Flateten flattens 28*28 into a single 784*1 input layer.
    # In short, input layer has 784 nodes vertically. Flatten to simply the structure of neural network vertically
    keras.layers.Flatten(input_shape=(28, 28)),

    # hidden layer is 128 deep nodes, relu return the value 0 means faster, tf tensor flow, nn neural network
    keras.layers.Dense(units=128, activation=tf.nn.relu),

    # output is 0-10, so we have 10 nodes in output (since input had 784 nodes and output has 10 nodes), softmax takes the greatest number out
    keras.layers.Dense(units=10, activation=tf.nn.softmax)
])

# Compile our model, optimizer for more correction or accuracy, loss tells how wrong we are
model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train our model, using our training data, epochs=5 optimizing in 5 steps and will help to reduce the loss/error. Will take all 60000 images
model.fit(train_images, train_labels, epochs=10)  # so, 60000 images for training

# Test our model, using our testing data
test_loss = model.evaluate(test_images, test_labels)  # so, 10000 images for testing

# Make predictions
predictions = model.predict(test_images)

# print out prediction
print(list(predictions[1]).index((max(predictions[1]))))  # prediction at 1

# Print the correct answer
print(test_labels[1])

print("Successful")