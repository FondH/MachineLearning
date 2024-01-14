import numpy as np
from SoftMax import *
from data_process import zero_padding
class Block:

    def __init__(self, name="Block"):
        self.name = name
        self.reg_rate = 0.05
        self.mu = 0
        self.sigma =  0.1
        self.mode = 'train'

    def forwards(self, x) -> any:
        pass

    def backwards(self, dout) -> any:
        pass

    def test_dims(self,x) -> None:
        pass

    def fix_mode(self, mode):
        self.mode = mode

class MLP(Block):
    def __init__(self, lname, dims, lr=0.001, bia=True):
        super().__init__(name="MLP-" + lname)
        self.lr = lr

        self.w = np.random.normal(self.mu,self.sigma, dims)

        #self.w = np.random.randn(*dims)
        if bia:
            self.bias = np.ones(dims[-1]) * 0.01
            # self.bia = np.zeros(dims[-1])

    # forwards:
    # input: [256, 120]  w: [120, 84]
    # return [256, 84]
    def forwards(self, x):
        self.input = x
        output = x @ self.w
        if hasattr(self, 'bia'):
            output += self.bia
        return output

    def backwards(self, dA):

        # dA [256, 84]  dw[120, 84] = [256,120].T @ [256,84]
        dW = self.input.T @ dA
        dA = dA @ self.w.T
        db = 0
        if hasattr(self, 'bia'):
            db = np.sum(dA, axis=1).mean()

        self.w -= self.lr * dW
        if hasattr(self, 'bia'):
            self.bia -= self.lr * db

        return dA

    def test(self, X):
        self.forwards(X)
        self.backwards(X)
        print()

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1/np.power(np.cosh(x),2)
def LeNet5_squash(x):
    return 1.7159*np.tanh(2*x/3)
def d_LeNet5_squash(x):
    return 1.14393*(1-np.power(tanh(2*x/3),2))

def ReLU(x):
    return np.where(x>0, x, 0)
def d_ReLU(x):
    return np.where(x>0, 1, 0)
class Sigmoid(Block):
    def __init__(self):
        super().__init__(name="Sigmoid")

    # Sigmoid函数
    # input[256,1,n,n]
    # return [256,1,n,n]
    def forwards(self, x):
        self.input = x
        #output = 1 / (1 + np.exp(-x))
        output = ReLU(x)
        return output

    def backwards(self, dout):
        #sigmoid_derivative = self.input * (1 - self.input)
        #dA = np.multiply(dout, sigmoid_derivative)
        dA = d_ReLU(dout)
        return dA
    
    def SLDM(self, dout):
        sigmoid_derivative = self.input * (1 - self.input)
        dA = dout * np.power(sigmoid_derivative,2) 

    def test(self):
        pass

class FFlatten(Block):
    def __init__(self):
        super().__init__(name="FFlatten")

    def forwards(self, x):
        self.input =  x[:, 0, 0, :]

        return self.input

    def backwards(self, dout):
        reverse_flatten = dout[:, np.newaxis, np.newaxis, :]


        return reverse_flatten



class Flatten(Block):
    def __init__(self):
        super().__init__(name="Flatten")

    def forwards(self, x):
        self.input_shape = x.shape
        output = x.reshape(x.shape[0], -1)
        return output/10

    def backwards(self, dout):
        dA = dout.reshape(self.input_shape)
        return dA

    def test(self):
        pass


class SoftMax(Block):
    def __init__(self):
        super().__init__(name="Softmax")

    def forwards(self, x):
        self.input = x
        self.output = softmax(x)
        return self.output

    def backwards(self, y):
        # 默认交叉熵
        dA = self.output
        dA[list(range(len(y))), y] -= 1
        # dA = (self.output.T) @ self.input
        return dA


class Model():
    def __init__(self, name, model_list):
        self.name = name
        self.model_list = model_list

    def output(self, x):
        for layer in self.model_list:
            x = layer.forwards(x)
        return x

    def fix_mode(self, mode):
        for layer in self.model_list:
            layer.fix_mode(mode)

    def backwards(self, y):
        dA = y
        for layer in self.model_list.__reversed__():
            dA = layer.backwards(dA)


class ConvolutionalLayer(Block):
    def __init__(self, lname, in_channels, out_channels, kernel_size, stride=1, padding=0, lr=0.005):
        super().__init__(name="ConvLayer" + lname)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.lr = lr

        mu, sigma = 0, 0.1
        # 初始化权重和偏置
        self.w = np.random.normal(mu, sigma, (kernel_size, kernel_size, in_channels, out_channels))
        self.bias = np.ones(out_channels) * 0.01
        #self.w = np.random.randn(kernel_size, kernel_size, in_channels, out_channels)
        #self.bias = np.zeros(out_channels)

    def _simple_conv(self, x):
        self.input = x

        batch, in_channel, in_h, in_w = x.shape
        out_h = (in_h + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (in_w + 2 * self.padding - self.kernel_size) // self.stride + 1
        assert (out_h == out_w and in_channel == self.in_channels)

        if self.padding > 0:
            X_padding = zero_padding(self.input, self.padding)
            # input_padding[:, :, self.padding:-self.padding, self.padding:-self.padding]=self.input
        else:
            X_padding = self.input

        output = np.zeros((batch, self.out_channels, out_h, out_w))

        for i in range(batch):
            for channel in range(self.out_channels):
                for h in range(out_h):
                    for w in range(out_w):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size

                        # window = x[i, :, h_start:h_end, w_start:w_end]
                        # output[i, channel, h, w] = np.sum(window * self.weights[channel]) + self.bias[channel]
                        conv_sum = 0
                        # 对input map 的所有channel 同一位置卷积结果平均
                        for m in range(in_channel):
                            window = X_padding[i, m, h_start:h_end, w_start:w_end]
                            conv_sum += np.sum(window * self.w[channel, m])

                        output[i, channel, h, w] = conv_sum + self.bias[channel]

        return output

    def _optimized_forward(self, x):
        self.input = x
        (m, n_H_prev, n_W_prev, n_C_prev) = x.shape
        (f, f, n_C_prev, n_C) = self.w.shape

        stride, pad = self.stride, self.padding

        n_H = int((n_H_prev + 2 * pad - f) / stride + 1)
        n_W = int((n_W_prev + 2 * pad - f) / stride + 1)

        # Initialize the output volume Z with zeros.
        Z = np.zeros((m, n_H, n_W, n_C))
        A_prev_pad = zero_padding(x, pad)
        for h in range(n_H):
            for w in range(n_W):
                # Use the corners to define the (3D) slice of a_prev_pad.
                A_slice_prev = A_prev_pad[:, h * stride:h * stride + f, w * stride:w * stride + f, :]
                # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron.
                Z[:, h, w, :] = np.tensordot(A_slice_prev, self.w, axes=([1, 2, 3], [0, 1, 2])) + self.bias

        assert (Z.shape == (m, n_H, n_W, n_C))

        # cache = (A_prev, W, b, hyper_parameters)
        return Z

    def forwards(self, x):
        return self._optimized_forward(x)

    def _simple_bac(self, dout):
        batch_size, _, in_height, in_width = self.input.shape
        _, _, out_height, out_width = dout.shape

        dW = np.zeros(self.w.shape)
        dX_padded = np.zeros((batch_size, self.in_channels, in_height + 2 * self.padding, in_width + 2 * self.padding))
        db = np.zeros(self.bias.shape)

        if self.padding > 0:
            X_padded = zero_padding(self.input, self.padding)
            # dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding] = self.input
        else:
            X_padded = self.input

        for i in range(batch_size):
            for j in range(self.out_channels):
                for k in range(self.in_channels):
                    for m in range(out_height):
                        for n in range(out_width):
                            h_start, w_start = m * self.stride, n * self.stride
                            window = X_padded[i, k, h_start:h_start + self.kernel_size,
                                     w_start:w_start + self.kernel_size]

                            # 更新权重梯度
                            dW[j, k] += window * dout[i, j, m, n]

                            # 更新输入梯度
                            dX_padded[i, k, h_start:h_start + self.kernel_size, w_start:w_start + self.kernel_size] += \
                                self.w[j, k] * dout[i, j, m, n]

        for j in range(self.out_channels):
            db[j] = np.sum(dout[:, j, :, :])

        if self.padding > 0:
            dX = dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dX = dX_padded


        return dX

    def _optimized_bac(self, dout):
        A_prev, W, b = self.input, self.w, self.bias
        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
        f, f, n_C_prev, n_C = W.shape
        m, n_H, n_W, n_C = dout.shape
        stride, pad = self.stride, self.padding

        dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
        dW = np.zeros((f, f, n_C_prev, n_C))
        db = np.zeros((1, 1, 1, n_C))

        if pad != 0:
            A_prev_pad = zero_padding(A_prev, pad)
            dA_prev_pad = zero_padding(dA_prev, pad)
        else:
            A_prev_pad = A_prev
            dA_prev_pad = dA_prev

        for h in range(n_H):
            for w in range(n_W):
                # Find the corners of the current "slice"
                vert_start, horiz_start = h * stride, w * stride
                vert_end, horiz_end = vert_start + f, horiz_start + f

                # Use the corners to define the slice from a_prev_pad
                A_slice = A_prev_pad[:, vert_start:vert_end, horiz_start:horiz_end, :]

                # Update gradients for the window and the filter's parameters
                dA_prev_pad[:, vert_start:vert_end, horiz_start:horiz_end, :] += np.transpose(
                    np.dot(W, dout[:, h, w, :].T), (3, 0, 1, 2))

                dW += np.dot(np.transpose(A_slice, (1, 2, 3, 0)), dout[:, h, w, :])
                db += np.sum(dout[:, h, w, :], axis=0)

        # Set dA_prev to the unpadded dA_prev_pad
        dA_prev = dA_prev_pad if pad == 0 else dA_prev_pad[:, pad:-pad, pad:-pad, :]

        # Making sure your output shape is correct
        assert (dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

        return dA_prev

    # dout[256 16 10 10]  -反卷积>  dw[16 5 5]   input [256 6 14 14]
    def backwards(self, dout):

        return self._optimized_bac(dout)


class PoolingLayer(Block):
    def __init__(self, pool_size, stride, mode='average'):
        super().__init__(name="PoolingLayer")
        self.pool_size = pool_size
        self.stride = stride
        self.mode = mode

    def _simple_for(self, x):
        batch_size, channels, height, width = x.shape
        out_height = (height) // self.stride
        out_width = (width) // self.stride

        output = np.zeros((batch_size, channels, out_height, out_width))

        for i in range(batch_size):
            for j in range(channels):
                for k in range(out_height):
                    for l in range(out_width):
                        h_start = k * self.stride
                        w_start = l * self.stride
                        window = x[i, j, h_start:h_start + self.pool_size, w_start:w_start + self.pool_size]

                        if self.mode == 'max':
                            output[i, j, k, l] = np.max(window)
                        elif self.mode == 'average':
                            output[i, j, k, l] = np.mean(window)

        return output

    def _optimized_for(self, x):
        m, n_H_prev, n_W_prev, n_C_prev = x.shape
        f, stride = self.pool_size, self.stride

        n_H = int(1 + (n_H_prev - f) / stride)
        n_W = int(1 + (n_W_prev - f) / stride)
        n_C = n_C_prev

        A = np.zeros((m, n_H, n_W, n_C))
        for h in range(n_H):
            for w in range(n_W):
                # Use the corners to define the current slice on the ith training example of A_prev, channel c
                A_prev_slice = x[:, h * stride:h * stride + f, w * stride:w * stride + f, :]
                # Compute the pooling operation on the slice. Use an if statement to differentiate the modes.
                if self.mode == "max":
                    A[:, h, w, :] = np.max(A_prev_slice, axis=(1, 2))
                elif self.mode == "average":
                    A[:, h, w, :] = np.average(A_prev_slice, axis=(1, 2))

        return A

    def forwards(self, x):
        self.input = x
        return self._optimized_for(x)

    def _simple_bac(self, dout):
        batch_size, channels, out_height, out_width = dout.shape
        # out_height, out_width = dout.shape[2], dout.shape[3]

        dX = np.zeros_like(self.input)

        for i in range(batch_size):
            for j in range(channels):
                for k in range(out_height):
                    for l in range(out_width):
                        h_start = k * self.stride
                        w_start = l * self.stride
                        window = self.input[i, j, h_start:h_start + self.pool_size, w_start:w_start + self.pool_size]

                        if self.mode == 'max':
                            max_val = np.max(window)
                            mask = (window == max_val)
                            dX[i, j, h_start:h_start + self.pool_size, w_start:w_start + self.pool_size] += mask * dout[
                                i, j, k, l]
                        elif self.mode == 'average':
                            average_val = dout[i, j, k, l] / (self.pool_size * self.pool_size)
                            dX[i, j, h_start:h_start + self.pool_size,
                            w_start:w_start + self.pool_size] += np.ones_like(window) * average_val

        return dX

    def _optimized_bac(self, dout):
        A_prev = self.input

        stride, f = self.stride, self.pool_size

        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape  # 256,28,28,6
        m, n_H, n_W, n_C = dout.shape  # 256,14,14,6

        dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))  # 256,28,28,6

        for h in range(n_H):
            for w in range(n_W):
                # Find the corners of the current "slice"
                vert_start, horiz_start = h * stride, w * stride
                vert_end, horiz_end = vert_start + f, horiz_start + f

                # Compute the backward propagation in both modes.
                if self.mode == "max":
                    A_prev_slice = A_prev[:, vert_start: vert_end, horiz_start: horiz_end, :]
                    A_prev_slice = np.transpose(A_prev_slice, (1, 2, 3, 0))
                    mask = A_prev_slice == A_prev_slice.max((0, 1))
                    mask = np.transpose(mask, (3, 2, 0, 1))
                    dA_prev[:, vert_start: vert_end, horiz_start: horiz_end, :] \
                        += np.transpose(np.multiply(dout[:, h, w, :][:, :, np.newaxis, np.newaxis], mask), (0, 2, 3, 1))

                elif self.mode == "average":
                    da = dout[:, h, w, :][:, np.newaxis, np.newaxis, :]  # 256*1*1*6
                    dA_prev[:, vert_start: vert_end, horiz_start: horiz_end, :] += np.repeat(np.repeat(da, 2, axis=1),
                                                                                             2, axis=2) / f / f

        assert (dA_prev.shape == A_prev.shape)
        return dA_prev

    def backwards(self, dout):
        return self._optimized_bac(dout)





class RBF(Block):
    def __init__(self, bitmap, name='RBF'):
        super().__init__(name=name)
        self.bitmap = bitmap

    def forwards(self, x):
        if self.mode == 'train':
            self.input_array =np.squeeze(x)
            return self.input_array

        elif self.mode == 'test':
            # (n_m,1,84) - n_m*[(10,84)] = (n_m,10,84)
            sub_weight = (self.input_array[:,np.newaxis,:] - np.array([self.bitmap]*self.input_array.shape[0])) # (n_m,10,84)
            y_hat = np.sum(np.power(sub_weight,2), axis=2) # (n_m, 10)

            return np.argmin(y_hat, axis=1) # (n_m,)

    def loss(self, label):
        self.bitmap_weight = self.bitmap[label, :]  # (n_m, 84) labeled version of weight
        loss = 0.5 * np.sum(np.power(self.input_array - self.bitmap_weight, 2), axis=1, keepdims=True)  # (n_m, )
        return np.sum(np.squeeze(loss))

    def backwards(self,label):
        #loss = self.loss(label)
        self.bitmap_weight = self.bitmap[label, :]  # (n_m, 84) labeled version of weight
        #loss = 0.5 * np.sum(np.power(self.input_array - self.bitmap_weight, 2), axis=1, keepdims=True)  # (n_m, )
        #return np.sum(np.squeeze(loss))
        dA = self.input_array - self.bitmap_weight    #(n_m, 84)
        return dA


class Flatten_dim(Block):
    def __init__(self, name='Flatten'):
        super().__init__(name=name)

    def forwards(self, x):
        #self.flatten = x[:,0,0,:]
        return x[:,0,0,:]

    def backwards(self, dout):
        reverse_flatten =dout[:, np.newaxis, np.newaxis, :]
        return reverse_flatten