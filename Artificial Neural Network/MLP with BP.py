# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 09:14:38 2019

@author: damien
"""

import numpy as np
import matplotlib.pyplot as plt
#from dataset import *
import time

def sigmoid(x):
#    return x
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
#    return x
    return (1 - sigmoid(x)) * sigmoid(x)

def cross_entropy(all_y_true, all_y_pred):
#    return np.sum(-(all_y_true * np.log(all_y_pred) + (1 - all_y_true) * np.log(1 - all_y_pred)))/343
    return -(all_y_true * np.log(all_y_pred) + (1 - all_y_true) * np.log(1 - all_y_pred)).mean()/100

def MSE(all_y_true, all_y_pred):
    return (np.sum((all_y_pred-all_y_true)**2))/343

def plot(x, y):
    plt.xlabel('Epoch')
    plt.ylabel('Loss value')
    plt.title('Regular MLP with 5 training samples')
    plt.plot(x,y,'r-',label="MLP")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',ncol=2, mode="expand", borderaxespad=0.)
    plt.show()
    
x = []
y = []

class MLP:
    # inputs: 64, hiddenlayer: n, output: 5  weights1=n*64 weights2=5*n x=64*1
    def __init__(self, weights1, weights2, bias1, bias2):
        self.w1 = weights1
        self.w2 = weights2
        self.b1 = bias1
        self.b2 = bias2

    # BackPropagation
    def ZH(self, x):
        return np.dot(x, np.transpose(self.w1)) + self.b1

    def H(self, x):
        return sigmoid(self.ZH(x))

    def ZY(self, x):
        h = self.H(x)
        return np.dot(h, np.transpose(self.w2)) + self.b2

    def Y(self, x):
        return sigmoid(self.ZY(x))

    # convert 2-d result into 1-d array
    def feedforwards(self, x):
        return np.reshape(sigmoid(self.ZY(x)) ,1)

    def batch_feedforwards(self, x):
        result = []
        for i in x: 
            result.append(self.feedforwards(i))
        return result
        
    def file_write(self):
      f = open("Conv. NN parameters.txt", "w")
      np.set_printoptions(threshold=1500)
      parameters = [self.w1, self.w2, self.b1, self.b2]      
      w1 = ','.join(map(str, parameters[0]))+'\n'*10
      w2 = ','.join(map(str, parameters[1]))+'\n'*10
      b1 = ','.join(map(str, parameters[2]))+'\n'*10
      b2 =  ','.join(map(str, parameters[3]))
                 
      f.write('w1= \n')
      f.write(w1)
      f.write('w2= \n')
      f.write(w2)
      f.write('b1= \n')
      f.write(b1)
      f.write('b2= \n')
      f.write(b2)
      f.close()
      
    def train(self, all_inputs, all_y_true, epoch, if_print):

        r = 0.6
        inputs = len(all_inputs[0])
        outputs = 1
        for epoch in range(epoch):

            for slice_inputs, slice_y_true in zip(all_inputs, all_y_true):
                
                slice_inputs = np.reshape(slice_inputs, (1, inputs))
                
                slice_y_true = np.reshape(slice_y_true, (1, outputs))
                
                # get output array  1 * 5
                slice_y_pred = self.Y(slice_inputs)
                
                # get hidden layer array 1 * m
                slice_h = self.H(slice_inputs)
                
                # for output layer
                delta_Y = (slice_y_pred - slice_y_true)
                dc_dw_y_h = np.dot(np.transpose(delta_Y), slice_h)
                dc_db_y = delta_Y

                # for hidden layer
                D = np.dot((slice_y_pred - slice_y_true), self.w2) * slice_h * (1 - slice_h)
                dc_dw_h_x = np.dot(np.transpose(D), slice_inputs)
                dc_db_h = D

                # update
                self.w1 -= r * dc_dw_h_x
                self.b1 -= r * dc_db_h
                self.w2 -= r * dc_dw_y_h
                self.b2 -= r * dc_db_y
                
            # depict epoch-cost graph
            
            if if_print == True: 
                print('Input', slice_inputs.shape, slice_inputs)
                print('True', slice_y_true.shape, slice_y_true)
                print('slice_y_pred', slice_y_pred)
                print('slice_h', slice_h)
                print('w1,b1,w2,b2',[self.w1, self.b1,self.w2, self.b2])

            if epoch % 1000 == 0:
#                fin = 0
                all_y_pred = np.apply_along_axis(self.feedforwards, 1, all_inputs)
                new_pred = []
                for i in all_y_pred:
                    new_pred.append(i[0])
                all_y_pred = np.array(new_pred)
                cost = MSE(all_y_true, all_y_pred)
                x.append(epoch)
                y.append(cost)
                print("epoch %d: %2.7f" % (epoch, cost))

### design the neural network with 6 hidden layers

def design_neural_network(inputs, hidden, outputs):
    print("let's begin")
    start = time.time()
    w1 = sigmoid(np.random.random((hidden, inputs)))
    w2 = sigmoid(np.random.random((outputs, hidden)))
    b1 = sigmoid(np.random.random((1, hidden)))
    b2 = sigmoid(np.random.random((1, outputs)))

    myneural = MLP(w1, w2, b1, b2)
    myneural.train(training_set,training_true, 10000, False) #5 training samples
    train_result = np.transpose(np.array(myneural.batch_feedforwards(testing_set)))
    true_result = np.array(testing_true)
    diff = np.average(np.abs(train_result-true_result)/true_result)
    Accuracy = (1-diff)*100
    print('Accuracy:', Accuracy)
    print('Predict result:', train_result[0][:10])
    print('True result:', true_result[:10])
    e_time = (time.time()-start)
    return e_time

'''
Main Program
'''
def multi_plot(hidden):
    cpu_time = []
    for i in hidden: 
        time = design_neural_network(3, i, 1)
        cpu_time.append(time)
    plt.plot(x[:10], y[:10])
    plt.plot(x[:10], y[10:20])
    plt.plot(x[:10], y[20:30])
    plt.plot(x[:10], y[30:40])
    plt.plot(x[:10], y[40:50])
    plt.legend(['2 neurons', '3 neurons', '4 neurons', '5 neurons', '6 neurons'], loc='upper right')
    print(cpu_time)
    
    
multi_plot([2,3,4,5,6])
