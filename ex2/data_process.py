import numpy as np
import struct
import os

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, f'{kind}-labels.idx1-ubyte')
    images_path = os.path.join(path, f'{kind}-images.idx3-ubyte')

    labels = read_idx(labels_path)
    images = read_idx(images_path)

    return images, labels


def data_preprocess(x,y):
    x[x<=40]=0
    x[x>40] =1
    return x,y

def normalize(image):
    image -= image.min()
    image = image / image.max()
    image = image * 1.275 - 0.1
    return image

def zero_padding(X, pad):
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=(0, 0))
    return X_pad

def plpot(loss_values, acc_values):
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
    plt.show()
    # plt.savefig('theta_1000_0.05.png')

