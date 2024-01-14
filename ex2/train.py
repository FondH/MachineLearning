from data_process import plpot
from SoftMax import *
from evaluate import *
from BitMap import loss
def train(model, x, y,ts,ty, iters, batch_size=128):
    # batch_size = 600
    loss_ls = []
    loss_ts = []
    acc_train = []
    acc_test = []

    # print(f"Epoch{-1}\n acc:{get_acc(x,y,model)}")
    from tqdm import trange
    for epoch in trange(iters):
        loss_sum = 0
        generator = batch_generator(x, y, batch_size)
        for i, (x_, y_) in enumerate(generator):
            # print(f"epoch:{i}:")
            y_hat = model.output(x_)
            #l = cross_entropy(y_hat, y_)
            l = loss(y_hat, y_)
            loss_sum = loss_sum + l
            model.backwards(y_)

            if not (((i + 1) * batch_size)) % 1000:
                loss_ls.append(l)
                l0, _ = model.output(x, y)
                l1, _ = model.output(ts,  ty)
                loss_ls.append(l0)
                loss_ts.append(l1)
        print(f"Epoch{epoch}\n Loss:{loss_sum / (60000)},acc:{cal_accuracy(x, y)}")

    #plpot(loss_ls, acc_train)
    return loss_ls, loss_ts, acc_train,acc_test