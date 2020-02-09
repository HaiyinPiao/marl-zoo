import torch
import math


def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
    return entropy.sum(1, keepdim=True)


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)


def set_init(layers):
    for layer in layers:
        torch.nn.init.xavier_normal_(layer.weight)
        # torch.nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.constant_(layer.bias, 0.0)
