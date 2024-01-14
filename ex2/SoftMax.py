import numpy as np
from BitMap import rbf_init_weight
def softmax(x):
    # x [6000,10]

    x_exp = np.exp(x)

    return x_exp / (np.sum(x_exp, axis=1, keepdims=True))





def cross_entropy(y_hat, y):
    if len(y.shape) > 1:
        y = np.squeeze(y, axis=1)
    # y_hat [6000.10] y[60000]
    p = np.log(y_hat[list(range(len(y))), y]).mean()
    return -p


def batch_generator(x, y, batch_size):
    num = len(x)
    for st in range(0, num, batch_size):
        ed = min(st + batch_size, num)
        yield x[st:ed], y[st:ed]



def get_gradient(x, y, y_hat, theta):
    reg_rate = 0.01
    l = loss(y_hat, y) + reg_rate * np.sum(theta * theta)
    y_hat[list(range(len(y))), y] -= 1
    gradient = (y_hat.T) @ x

    return gradient / len(x), l

