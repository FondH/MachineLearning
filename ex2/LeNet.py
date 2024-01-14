import torch
import numpy as np
from torch import nn
from d2l import  torch as d2l

from Fond_model import *
from data_process import load_mnist,normalize
from train import train
from evaluate import get_acc
#完整代码见仓库：
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1,6,kernel_size=5),nn.ReLU(),
                                 nn.AvgPool2d(kernel_size=2,stride=2),
                                 nn.Conv2d(6,16,kernel_size=5),nn.Sigmoid(),
                                 nn.AvgPool2d(kernel_size=2,stride=2),
                                 nn.Flatten(),
                                 nn.Linear(16*5*5, 120),nn.Sigmoid(),
                                 nn.Linear(120,84),nn.Sigmoid(),
                                 nn.Linear(84,10))

        def _initial(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.xavier_uniform_(m.weight)
        self.net.apply(_initial)
    def forward(self, X):
        return self.net(X)

 def train(model, train_iter, test_iter, lr, epochs, device):

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss = nn.CrossEntropyLoss()
        drawer = Drawer(xlabel='epoch', xlim=[0, epochs],
                        legend=['train loss', 'train acc', 'test acc'])
        timer = d2l.Timer()
        model.to(device)
        print(f'---------  {model.__class__.__name__} is training on {device} ----------')
        num_batches = len(train_iter)
        for epoch in range(epochs):
            metric = d2l.Accumulator(3)
            model.train()
            for i, (X, y) in enumerate(train_iter):
                optimizer.zero_grad()
                timer.start()
                X, y = X.to(device), y.to(device)
                X = X.type(torch.cuda.FloatTensor)
                y_hat = model(X)
                l = loss(y_hat, y)
                l.backward()
                optimizer.step()
                with torch.no_grad():
                    metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])

                timer.stop()
                train_loss = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]

                if i % (num_batches // 5) == 0 or i == num_batches - 1:
                    drawer.add(epoch + (i + 1) / num_batches, (train_loss, train_acc, None))
                # drawer.add(epoch + (i + 1) / num_batches, (train_loss, train_acc, None))
            test_acc = get_test_acc(model, test_iter, device)
            print(f'epoch:{epoch}的loss: {train_loss}    \nacc:{train_acc}   test_acc:{test_acc}')
            # animator.add(epoch + 1, (None, None, test_acc))
            drawer.add(epoch + 1, (None, None, test_acc))

            # is_save = False
            if epoch == epochs - 1:
                is_save = True
                drawer.draw(name=model.__class__.__name__, Is_save=is_save)
        print(f'用时: {timer.sum()}s')
        if timer.sum():
            print(f'speed:{metric[2] * epochs / (timer.sum()):.2f} sample/sec')

        # dump_model(model, f'训练时长: {timer.sum():.1f} sec\n'
        #                              f'train_loss: {train_loss}    acc:{train_acc}   test_acc:{test_acc}')

 def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

class TensorDataset(Dataset):
    """
    TensorDataset继承Dataset, 重载了__init__(), __getitem__(), __len__()
    实现将一组Tensor数据对封装成Tensor数据集
    能够通过index得到数据集的数据，能够通过len，得到数据集大小
    """

    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
            return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.shape[0]



if __name__ == "__main__":

    data_path = 'G:\大三上\机器学习\code\ex2\mnist_data_handwritten'
    train_images, train_labels = load_mnist(data_path, kind='train')

    test_images, test_labels = load_mnist(data_path, kind='t10k')
    x, y = normalize(zero_padding(train_images[:, np.newaxis, :, :], 2)[:60000]), train_labels[:60000]
    x = torch.tensor(x, dtype=torch.float)

    tx, ty = normalize(zero_padding(test_images[:, np.newaxis, :, :], 2)[:60000]), test_labels[:60000]
    tx = torch.tensor(tx, dtype=torch.float)
    train_iter = data.DataLoader(TensorDataset(x, y), 256, shuffle=True, num_workers=4)
    test_iter = data.DataLoader(TensorDataset(tx, ty), 256, shuffle=False, num_workers=4)

    model_lis = [LeNet()]
    # train_iter = [(torch.randn((10,1,224,224)),torch.arange(10)) for _ in range(10)]
    # test_iter = [(torch.randn((8,1,224,224)), torch.arange(8)) for _ in range(10)]
    epoch = 20
    lr = [0.005]
    device = d2l.try_gpu()
    for (lr, model) in zip(lr, model_lis):
        train(model, train_iter, test_iter, lr, epoch, device)
