import numpy as np


def predict(test_images, model):
    scores = model.output(test_images)
    preds = np.argmax(scores, axis=1)
    return preds


def cal_accuracy(y_pred, y):
    # TODO: Compute the accuracy among the test set and store it in acc
    rs = 0
    for _ in range(len(y)):

        if y_pred[_] == y[_]:
            rs += 1
    return rs / len(y)


def get_acc(x, y, model):
    preds = predict(x, model)
    return cal_accuracy(preds, y)


