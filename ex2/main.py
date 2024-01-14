from Fond_model import *
from data_process import load_mnist,normalize
from train import train
from evaluate import get_acc
import numpy as np
from BitMap import rbf_init_weight
if __name__ == '__main__':

    data_path = 'mnist_data_handwritten'
    alphals = [5e-5, 5e-5,5e-5,2e-4]
    alpha = 1e-3

    layers = [ConvolutionalLayer("1->6",1,6,5,stride=1,padding=0,lr=alphals[0]),
              Sigmoid(),
              PoolingLayer(2,2,mode='average'),
              ConvolutionalLayer("6->16",6,16,5,stride=1,padding=0,lr=alphals[1]),
              Sigmoid(),
              PoolingLayer(2,2,mode='average'),
              #Flatten(),
              ConvolutionalLayer("16->120",16,120,5,stride=1,padding=0,lr=alphals[2]),
              #MLP("400->120",(400,120),lr=alpha),
              Sigmoid(),
              Flatten_dim(),
              MLP("120->84",(120,84),lr=alphals[3]),
              Sigmoid(),
              #MLP("84->10",(84,10),lr=alpha),
              #SoftMax(),
              RBF(rbf_init_weight()),
             ]

    LeNet5 = Model(name="Fond-LeNet",model_list=layers)

    train_images, train_labels = load_mnist(data_path, kind='train')
    test_images, test_labels = load_mnist(data_path, kind='t10k')

    x,y = normalize(zero_padding(train_images[:, :, :, np.newaxis], 2)[:60000]),  train_labels[:60000]
    tx,ty = normalize(zero_padding(test_images[:, :, :, np.newaxis], 2)[:60000]),  test_labels[:60000]
    #x = np.expand_dims(x, axis=-1)
    #iter = batch_generator(train_images,train_labels,batch_size=256)
    train(LeNet5,x,y,tx,ty,iters=30,batch_size=256)
    print("Test Acc",get_acc(tx,ty,LeNet5))
    









































    
