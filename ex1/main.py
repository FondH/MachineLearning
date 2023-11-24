# coding=utf-8
import numpy as np
import struct
import os

from data_process import load_mnist, load_data
from train import train
from evaluate import predict, cal_accuracy
    

if __name__ == '__main__':
    # initialize the parameters needed
    root =os.getcwd()
    mnist_dir = os.path.join(root,"mnist_data")
    train_data_dir = "train-images-idx3-ubyte"
    train_label_dir = "train-labels-idx1-ubyte"
    test_data_dir = "t10k-images-idx3-ubyte"
    test_label_dir = "t10k-labels-idx1-ubyte"
    k = 10
    iters = 100
    alpha = 0.0005



    # get the data
    train_images, train_labels, test_images, test_labels = load_data(mnist_dir, train_data_dir, train_label_dir, test_data_dir, test_label_dir)
    print("Got data. ") 

    import time
    # train the classifier
    start = time.time()
    theta = train(train_images, train_labels, k, iters, alpha,batch_size=256)

    print("Finished training. ")
    print(f"Sum of Time:{time.time()-start} s") 

    # evaluate on the testset
    y_predict = predict(test_images, theta)
    accuracy = cal_accuracy(y_predict, test_labels)

    import pickle

    with open('./theta_1000_0.05.p', 'wb') as f:
        pickle.dump(theta, f)
    print("Test Acc",accuracy)
    print("Finished test. ") 

    
    
