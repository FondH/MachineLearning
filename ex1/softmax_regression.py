# coding=utf-8
import numpy as np
from evaluate import predict, cal_accuracy, get_acc
def softmax(x):
    # x [6000,10]

    x_exp = np.exp(x)

    return x_exp / (np.sum(x_exp,axis=1,keepdims=True))
    
def loss(y_hat,y):
    if len(y.shape) >1 :
        y = np.squeeze(y,axis=1)
    #y_hat [6000.10] y[60000]
    p = np.log(y_hat[list(range(len(y))),y]).mean()
    return -p

def batch_generator(x,y,batch_size):
    num = len(x)
    for st in range(0,num,batch_size):
        ed = min(st+batch_size,num)
        yield x[st:ed], y[st:ed]

def plpot(loss_values,acc_values):
    import matplotlib.pyplot as plt
    # 创建新的图表窗口
    plt.figure()

    # 绘制 loss 曲线
    plt.subplot(2, 1, 1)  # 创建一个 2x1 的图表网格，并选择第一个子图
    plt.plot(loss_values, label='Loss')
    plt.title('Loss over time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制 accuracy 曲线
    plt.subplot(2, 1, 2)  # 创建一个 2x1 的图表网格，并选择第二个子图
    plt.plot(acc_values, label='Accuracy', color='orange')
    plt.title('Accuracy over time')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    
    # 显示图表
    plt.tight_layout()  # 自动调整子图参数，以保证图表不会重叠
   # plt.show()
    plt.savefig('theta_1000_0.05.png')


def get_gradient(x,y,y_hat,theta):
    reg_rate = 0.01
    l = loss(y_hat, y) +  reg_rate * np.sum(theta*theta)
    y_hat[list(range(len(y))),y]-=1
    gradient = (y_hat.T) @ x 

    return gradient/len(x) ,l
def softmax_regression(theta, x, y, iters, alpha,batch_size=1):
    # TODO: Do the softmax regression by computing the gradient and 
    # the objective function value of every iteration and update the theta
    # softmax: np.exp(x[i])/np.exp(x).sum()  
    # loss: min(-y_i * logy_hat)
    # gradient:  softmax(xi)-yi
    #x[60000,784] y[60000,10] theta[10,784]
    if len(y.shape)>1:
        y=np.squeeze(y)
    #batch_size = 600
    loss_ls = []
    acc_train = []
    acc_test = []

    print(f"Epoc:-1\n  Acc:{get_acc(x, y, theta)}")
    from tqdm import trange
    for epoch in range(iters):
        loss_sum = 0
        genertor = batch_generator(x, y, batch_size)
        for i,(x_,y_) in enumerate(genertor):
            #print(f"epoch:{i}:")
            y_hat = softmax(x_ @ theta.T)
            gradient,l = get_gradient(x_,y_,y_hat,theta)
            #print(l)
            loss_sum = loss_sum+l
            theta -= alpha * (gradient - 0.01*theta) 

            
            if not (((i+1)*batch_size))%1000:
                loss_ls.append(l.copy())
                acc_train.append(get_acc(x, y, theta))
                #print(f"Loss:{l}")
        print(f"Epoch{epoch}\n Loss:{loss_sum/(60000)} Acc:{get_acc(x,y,theta)}")

    plpot(loss_ls,acc_train)
    return theta
    

def softmax_regression_base(theta, x, y, iters, alpha):
    m, n = x.shape
    k = theta.shape[0]
    loss_ls = []
    acc_train = []
    acc_test = []
    from tqdm import trange
    for iteration in range(iters):
        cost = 0
        acc = 0
        grad = np.zeros_like(theta)

        for i in range(m):
            # 计算 softmax 概率
            logits = np.dot(theta, x[i])
            probabilities = np.exp(logits) / np.sum(np.exp(logits))

            # 计算损失
            cost -= np.log(probabilities[y[i]])

            # 计算梯度
            for j in range(k):
                indicator = 1 if y[i] == j else 0
                grad[j, :] += (probabilities[j] - indicator) * x[i]



        # 计算平均损失和梯度
        cost /= m
        grad /= m
        acc = get_acc(x,y,theta)
        loss_ls.append(cost.copy())
        acc_train.append(acc)
        # 更新 theta
        theta -= alpha * grad

        print(f"Epoch {iteration}: Loss:{cost} Acc:{acc}")
    plpot(loss_ls,acc_train)
    return theta