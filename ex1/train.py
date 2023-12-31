# coding=utf-8
import numpy as np

from data_process import data_convert,data_preprocess
from softmax_regression import softmax_regression, softmax_regression_base


def train(train_images, train_labels, k, iters = 5, alpha = 0.5,batch_size=100):
    #k=10 n 784 m 60000
    m, n = train_images.shape
    # data processing
    x, y = data_preprocess(train_images, train_labels, m, k) # x:[m,n], y:[1,m]
    # x[]
    # Initialize theta.  Use a matrix where each column corresponds to a class,
    # and each row is a classifier coefficient for that class.
    theta = np.random.rand(k, n) # [k,n][10,784] [60000 , 784]
    # do the softmax regression
    theta = softmax_regression(theta, x, y, iters, alpha)
    return theta

