import torch

batch_size = 100


class_num = 10

def torch_one_hot(labels):

    one_hot = torch.zeros(batch_size, class_num).scatter_(1, labels.reshape(batch_size, 1), 1)

    return one_hot
