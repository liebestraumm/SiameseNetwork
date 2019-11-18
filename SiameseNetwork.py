#----- IFN680 Assignment 2 -----------------------------------------------#
#  Siamese network
#
#    Student no: n9837809
#    Student name: CARLOS ARTURO AGELVIS VERGARA
#-------------------------------------------------------------------------#

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, MaxPooling2D, Activation, Flatten, Dense, Dropout, Lambda
from keras.layers import concatenate
from keras.optimizers import Adadelta, adam
import random
import tensorflow as tf
from tensorflow import keras
from sklearn import model_selection
from keras import backend as K
from keras.optimizers import RMSprop

##########Global variables##################
epochs = 1              #To be used by training the model. This is the number of epochs or cycles
batch_size = 128        #To be used by training the model. This is the size of the batch used in each epoch
n_pairs = 20            #To be used by verifying pairs in verifypairs() method
margin = 1              #To be used by the contrastive_loss function

def create_pairs(x, indices, length):
    """
    Function: create_pairs
    Parameters:
        - x: training data containing the images
        - indices: The possitions of the array where the images are
        - length: length of the array containing the labels
    Returns: 2 numpy arrays.
        - One containing the image pairs
        - Another one containing the respective label 
    Description: Creates positive and negative pairs with their respective labels: 
                 0 = positive, 1 = negative
    """
    pairs = []
    labels = []
    n = min([len(indices[d]) for d in range(length)]) - 1
    for d in range(length):
        for i in range(n):
            z11, z21 = indices[d][i], indices[d][i + 1]
            inc = random.randrange(1, length)
            dn = (d + inc) % length
            z12, z22 = indices[d][i], indices[dn][i]
            #positive pair
            pairs += [[x[z11], x[z21]]]
            labels.append(0)
            #negative pair
            pairs += [[x[z12], x[z22]]]
            labels.append(1)
    return np.array(pairs), np.array(labels)

def create_base_network(input_shape):
    """
    Function: create_base_network
    Parameters:
        - input_shape: The shape of the image (in this case 28,28)
    Returns: CNN model 
    Description: Creates the CNN architecture
    """
    input_img = Input(shape=input_shape)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape = input_shape)(input_img)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(90)(x)
    return Model(input_img, x)

def euclidean_distance(vects):

    """
    Function: euclidean_distance
    Parameters: 
        - vects: x and y arrays representing the processed images by the CNN model
    Returns: a float number representing the euclidean distance between both images
    Description: Calculates euclidean distance
    """
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    """
    Function: eucl_dist_out
    Parameters: 
        - shapes: shapes of the images
    Returns: The shape of the image
    """
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y, y_pred, margin = 1):
    """
    Task 2: Loss function implementation
    Function: contrastive_loss
    Parameters:
        - y: positive labels (0)
        - y_pred: prediction of the model
        - margin: number that determines how far the embeddings of negative pair are pushed apart
    Returns: a float number indicating the loss
    Description: Implements the contrastive loss function
    """
    Y0 = 1 - y
    Y1 = y
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean((Y0 * square_pred) + (Y1 * margin_square))

def compute_accuracy(y_true, y_pred):
    """
    Function: compute_accuracy
    Parameters:
        - y: positive labels (0)
        - y_pred: prediction of the model
    Returns: a float number indicating the accuracy of the model 
    Description: Computes classification accuracy with a fixed threshold on distances.
    """
    pred = y_pred.ravel() > 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    """
    Function: compute_accuracy
    Parameters:
        - y: positive labels (0)
        - y_pred: prediction of the model
    Returns: a float number indicating the accuracy of the model 
    Description: Computes classification accuracy with a fixed threshold on distances.
    """
    return K.mean(K.equal(y_true, K.cast(y_pred > 0.5, y_true.dtype)))


def contrasive_loss_test():
    """
    Function: contrastive_loss_test
    Parameters:
        - None
    Returns: This function doesn't return anythin
    Description: Tests the contrasive_loss function, computing in both numpy and tensorflow.
                 If both results from numpy and tensorflow are the same, the function is correct.
    """
    global margin
    num_data = 10
    feat_dim = 6
    
    embeddings = [np.random.rand(num_data, feat_dim).astype(np.float32),
                  np.random.rand(num_data, feat_dim).astype(np.float32)]
    labels = np.random.randint(0, 1, size=(num_data)).astype(np.float32)
    
    loss_np = 0.
    x = embeddings[0]
    y = embeddings[1]
    
    """
    Compute loss with numpy
    """
    for i in range(num_data):
        Y0 = 1 - labels[i]
        Y1 = labels[i]
        sum_square = np.sum(np.square(x[i] - y[i]))
        y_pred = np.sqrt(max(sum_square, K.epsilon())) 
        square_pred = np.square(y_pred)
        margin_square = np.square(max(margin - y_pred, 0.))
        loss_np += ((Y0 * square_pred) + (Y1 * margin_square))
    loss_np /= num_data
    print("Contrastive loss with numpy: ", loss_np)
    
    """
    Compute loss with tensor
    """
    with tf.Session() as sess:
        euclidean = euclidean_distance([x,y])
        loss_tf = contrastive_loss(labels, euclidean, margin)
        loss_tf_val = sess.run(loss_tf)
        print('Contrastive loss computed with tensorflow', loss_tf_val)

def verify_pairs(pairs, label, npairs):
    """
    Function: verifypairs
    Parameters:
        - pairs: the image pairs
        - label: the target label (0 for positive pair, 1 for negative pair)
        - npairs: The number of pairs desired to be verified
    Description: Verifies that the pairs have their correct label by showing the image pairs
                 and their respective label.
    """
    for i in range(npairs):
        f = pairs[i]
        f = np.squeeze(pairs[i])
        for j in range(len(f)):
            plt.figure(figsize=(2,2))
            if j == 0:
                if label[i] == 0:
                    text = "Positive Pair. Label {0}".format(label[i])
                    plt.title(text)
                else:
                    text = "Negative Pair. Label {0}".format(label[i])
                    plt.title(text)
            plt.imshow(f[j])
            plt.show()
    plt.close
    
"""
Task 1a: Create dataset
"""
#labels to be used only in training, validation and testing
req = [0,1,2,4,5,9]  
#Labels to be used only in testing

not_req = [3,6,7,8]   #Labels to be used only in testing

#Loading the whole dataset
(x_train, y_train) , (x_test, y_test) = fashion_mnist.load_data()

#Training labels from the 6 classes
Y_filtered_tr = np.array([y for y in y_train if y in req])

#Test labels from the 4 classes 
Y_filtered_te = np.array([y for y in y_test if y not in req]) 
 
#Training images from the 6 classes
X_filtered_tr = np.array([x_train[i] for i in range (len(x_train)) if y_train[i] in req])

#Test images from the 4 classes
X_filtered_te = np.array([x_test[i] for i in range (len(x_test)) if y_test[i] not in req])

#Splits training and test datasets of classes ("top", "trouser", "pullover", "coat",
#                                               "sandal","ankle boot") (80% training, 20% test)
num_training = int((X_filtered_tr.shape[0])*0.8)  
x_train2, x_test2 = np.array(X_filtered_tr[:num_training]), np.array(X_filtered_tr[num_training:])
y_train2,y_test2 = np.array(Y_filtered_tr[:num_training]), np.array(Y_filtered_tr[num_training:])

# reshape the input arrays to 4D (batch_size, rows, columns, channels) for 
img_rows, img_cols = x_train2.shape[1:3]
x_train2 = x_train2.reshape(x_train2.shape[0], img_rows, img_cols, 1)
x_test2 = x_test2.reshape(x_test2.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train2 = x_train2.astype('float32')
x_test2 = x_test2.astype('float32')
x_train2 /= 255
x_test2 /= 255
input_shape = x_train2.shape[1:]

#Creates indices for 2nd training set (80% of training data)
indices = [np.where(y_train2 == i)[0] for i in req]

#Creates positive and negative training pairs with their labels
tr_pairs, tr_y = create_pairs(x_train2, indices, len(req))

#Creates indices for 2nd test set (20% of training data)
indices = [np.where(y_test2 == i)[0] for i in req]

#Creates positive and negative test pairs with their labels for test 1
te_pairs, te_y = create_pairs(x_test2, indices, len(req))

#Splits training and test datasets of classes ("dress", "sneaker", "bag", "shirt") 
#                                                  (only for evaluation test)
#3rd Test labels for Test 3
y_test3 = Y_filtered_te

#3rd Test images for Test 3
x_test3 = X_filtered_te

# reshape the input arrays to 4D (batch_size, rows, columns, channels)
img_rows, img_cols = x_test3.shape[1:3]
x_test3 = x_test3.reshape(x_test3.shape[0], img_rows, img_cols, 1)
x_test3 = x_test3.astype('float32')
x_test3 /= 255
input_shape = (img_rows, img_cols, 1)

#Creates indices for 3rd test set (classes ("dress", "sneaker", "bag", "shirt"))
indices2 = [np.where(y_test3 == i)[0] for i in not_req]
#Creates 2nd positive and negative test pairs with their labels for test 3
te2_pairs, te2_y = create_pairs(x_test3, indices2, len(not_req))

#4rd Test images. Concatenates pairs from test 1 and 3 to make Test 2
x_test4 = np.concatenate((te_pairs, te2_pairs), axis=0)

#4rd Test labels for Test 2. Concatenates pairs from test 1 and 3 to make Test 2
y_test4 = np.concatenate((te_y, te2_y), axis = 0)
"""
Task 1b: Verify dataset by showing pairs with their respective lables (Label 0: Similar images, Label 1: Different Images)
"""
print("PAIRS FROM THE TRAINING DATA(6 classes)")
verify_pairs(tr_pairs, tr_y, n_pairs)
print("PAIRS FROM THE TEST DATA(6 classes)")
verify_pairs(te_pairs, te_y, n_pairs)
###########################################################################################
"""
Task 2: Test the contrastive loss function
"""
contrasive_loss_test()
######################################################################################

"""
Task 3: Build the siamese network
"""
#Creates the base network to be shared
base_network = create_base_network(input_shape)
#
input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)   
        
#Inputs img 1 to the CNN
processed_a = base_network(input_a)   

#Inputs img 2 to the CNN     
processed_b = base_network(input_b)

#Distance value to be output in the final layer of CNN
distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

#Inputs images to the siamese model
model = Model([input_a, input_b], distance)

#Shows summary of the model
model.compile(loss=contrastive_loss, optimizer=Adadelta(), metrics=[accuracy])
model.summary
###############################################################################

"""
Task 4: Training of the network and plot of accuracy
"""

#Training of the model
siamese_model = model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

#Plotting Training vs. Test
plt.plot(siamese_model.history['accuracy'])
plt.plot(siamese_model.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#Plotting Loss
plt.plot(siamese_model.history['loss'])
plt.plot(siamese_model.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

"""
Task 5: Evaluate generalization capability of the model
"""
#Test 1: Only using 6 class dataset for testing
y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(tr_y, y_pred)
y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(te_y, y_pred)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

#Passing test pairs te_pairs and the label test pairs te_y
score = model.evaluate([te_pairs[:, 0], te_pairs[:, 1]], te_y, verbose=0)
print('Test loss -- Test 1:', score[0])
print('Test accuracy -- Test 1: {0}%'.format(score[1]*100))

#Test 2: Concatenate both 2nd and 3rd testing pairs for evaluation
#Passing test pairs x_test4 and the label test pairs y_test4
score = model.evaluate([x_test4[:, 0], x_test4[:, 1]], y_test4, verbose=0)

print('Test loss -- Test 3:', score[0])
print('Test accuracy -- Test 3: {0}%'.format(score[1]*100))

#Test 3: Only using the 4 class dataset for testing
#Passing test pairs te2_pairs and the label test pairs te2_y
score = model.evaluate([te2_pairs[:, 0], te2_pairs[:, 1]], te2_y, verbose=0)
print('Test loss -- Test 2:', score[0])
print('Test accuracy -- Test 2: {0}%'.format(score[1]*100))
#####################################################################################

