#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 19:33:48 2019

@author: kZaf
"""
import time

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import matplotlib.pyplot as plt

tfe.enable_eager_execution()

# Introducing 10000 data pairs
NUM_EXAMPLES = 10000

# Define input x and output y introducing noise 
x = tf.random_normal([NUM_EXAMPLES]) 
noise = tf.random_normal([NUM_EXAMPLES])
y = x * 3 + 2 + noise

# Create variables for weight (W) and bias (b)
W = tf.Variable(0.)
b = tf.Variable(0.)

# Various loss functions
# (source: https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html)
def Huber(y, y_predicted, delta=1.0):
    error = y_predicted - y
    linear = (delta * tf.abs(error) - 0.5 * delta**2)
    quadratic = (0.5 * tf.square(error))
    return tf.reduce_mean(tf.where(tf.abs(error) <= delta, quadratic, linear))

def L1(y, y_predicted):
    error = y_predicted - y
    return tf.reduce_mean(tf.abs(error))

def L2(y, y_predicted):
    error = y_predicted - y
    return tf.reduce_mean(tf.square(error))

def hybrid(y, y_predicted):
    error = y_predicted - y
    return tf.reduce_mean(tf.abs(error)) + tf.reduce_mean(tf.square(error))

# Define the linear predictor.
def prediction(x):
  return x * W + b

def gradients(train_steps, learning_rate, loss_fn, W, b):
    # Calculate gradients
    for i in range(train_steps):
      
      # Watch the gradient flow 
      with tf.GradientTape() as tape:
        
        # Forward pass 
        yhat = prediction(x)
        
        # Calcuate the loss -choose from Huber, L1, L2
        loss = loss_fn(y, yhat)
      
      # Evalute the gradient with respect to the parameters
      dW, db = tape.gradient(loss, [W, b])
    
      # Update the parameters using Gradient Descent  
      W.assign_sub(dW * learning_rate)
      b.assign_sub(db * learning_rate)
    
      # Print the loss every 500 iterations 
      if i % 500 == 0:
        print("Loss at step {:03d}: {:.3f}".format(i, loss))
          
    print(f'W : {W.numpy()} , b  = {b.numpy()} ')
    
    return loss.numpy()


def run(train_steps, learning_rate, W_init=0, b_init=0):
    # Initializations
    W.assign(W_init)
    b.assign(b_init)
    
    # Calculate gradients - loss L1
    loss = gradients(train_steps, learning_rate, L1, W, b)
    # Plot display
    plt.figure()
    plt.plot(x, y, 'bo', label='Original data')
    plt.plot(x, prediction(x), 'r', label='Prediction w/ L1 loss function')
    plt.title('Linear Regression Result after ' + str(train_steps) + r' epochs and $\alpha$=' + str(learning_rate) + '\nFinal loss: ' + str(loss))
    plt.legend()
    #plt.show()
    plt.savefig('steps' + str(train_steps) + 'LR' + str(learning_rate) + 'lossL1.png')
    
    # Initializations
    W.assign(W_init)
    b.assign(b_init)
    
    # Calculate gradients - loss L2
    loss = gradients(train_steps, learning_rate, L2, W, b)
    # Plot display
    plt.figure()
    plt.plot(x, y, 'bo', label='Original data')
    plt.plot(x, prediction(x), 'r', label='Prediction w/ L2 loss function')
    plt.title('Linear Regression Result after ' + str(train_steps) + r' epochs and $\alpha$=' + str(learning_rate) + '\nFinal loss: ' + str(loss))
    plt.legend()
    #plt.show()
    plt.savefig('steps' + str(train_steps) + 'LR' + str(learning_rate) + 'lossL2.png')
    
    # Initializations
    W.assign(W_init)
    b.assign(b_init)
    
    # Calculate gradients - loss Huber
    loss = gradients(train_steps, learning_rate, Huber, W, b)
    # Plot display
    plt.figure()
    plt.plot(x, y, 'bo', label='Original data')
    plt.plot(x, prediction(x), 'r', label='Prediction w/ Huber loss function')
    plt.title('Linear Regression Result after ' + str(train_steps) + r' epochs and $\alpha$=' + str(learning_rate) + '\nFinal loss: ' + str(loss))
    plt.legend()
    #plt.show()
    plt.savefig('steps' + str(train_steps) + 'LR' + str(learning_rate) + 'lossHuber.png')
    
    # Initializations
    W.assign(W_init)
    b.assign(b_init)
    
    # Calculate gradients - loss Hybrid
    loss = gradients(train_steps, learning_rate, hybrid, W, b)
    # Plot display
    plt.figure()
    plt.plot(x, y, 'bo', label='Original data')
    plt.plot(x, prediction(x), 'r', label='Prediction w/ L1+L2 loss function')
    plt.title('Linear Regression Result after ' + str(train_steps) + r' epochs and $\alpha$=' + str(learning_rate) + '\nFinal loss: ' + str(loss))
    plt.legend()
    #plt.show()
    plt.savefig('steps' + str(train_steps) + 'LR' + str(learning_rate) + 'lossHybrid.png')

if __name__ == '__main__':
    run(3000, 0.01)
    run(2000, 0.01)
    run(1000, 0.01)
    run(3000, 0.001)
    run(2000, 0.001)
    run(1000, 0.001)
    run(1000, 0.0001, 2.5, 1.5)
    run(1000, 0.0001, 3.5, 2.5)

    
    