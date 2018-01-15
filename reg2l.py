import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_cn(weight):
    u, s, v = weight.svd()
    return torch.max(s) / (torch.min(s) + 1E-6)

def compute_cn_reg(weight, alpha):
    u, s, v = weight.svd()
    alpha = F.sigmoid(alpha)
    s = alpha * torch.mean(s)
    return weight + torch.matmul(u * s, v.permute(1, 0))

class Regularizer(nn.Module):
    def __init__(self, alpha=5E-2, learnable=False):
        alpha = -torch.log(1 / torch.Tensor([alpha]) - 1)
        self.alpha = nn.Parameter(alpha, requires_grad=False)
        self.learnable = learnable

    def train(self, mode=True):
        if not self.learnable:
            return
        self.alpha.requires_grad = not mode

    def forward(self, x):
        raise NotImplementedError

class LinearCNRegularizer(Regularizer):
    def __init__(self, linear_module, **kwargs):
        super().__init__(**kwargs)
        self.lin_module = linear_module
        self.weight = self.lin_module.weight
        self.bias = self.lin_module.bias

    def forward(self, x):
        weight = compute_cn_reg(self.weight, self.alpha)
        return F.linear(x, weight, self.bias)

def Conv2dCNRegularizer(Regularizer):
    def __init__(self, conv2d_module, **kwargs):
        super().__init__(**kwargs)
        self.conv_module = conv2d_module
        self.weight = conv2d_module.weight

    def forward(self, x):
        weights = [w.squeeze(0) for w in self.weight.split(1, 0)]
        weights = [[w0.squeeze(0) for w0 in w1.split(1, 0)] for w1 in weights]
        regularized = []
        for weight_list in weights:
            weight_slice = []
            for weight in weight_list:
                weight_slice.append(compute_cn_reg(weight, self.alpha))
            regularized.append(weight_slice)
        regularized = torch.stack([torch.stack(w) for w in regularized])
        c = self.conv_module
        return F.conv2d(x, regularized, c.bias, c.stride, c.padding, c.dilation, c.groups)

