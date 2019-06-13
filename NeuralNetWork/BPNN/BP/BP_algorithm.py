import torch
from torch.autograd import Variable
import numpy as np

_input = torch.Tensor([[0.05], [0.10]])
_ty = torch.Tensor([[0.01], [0.99]])  # standrd output
_b1 = torch.Tensor([0.35])
_b2 = torch.Tensor([0.60])
##########################3
w1 = Variable(torch.Tensor([[0.15, 0.2], [0.25, 0.30]]), requires_grad=True)
w2 = Variable(torch.Tensor([[0.40, 0.45], [0.5, 0.55]]), requires_grad=True)
h1 = w1.mm(_input) + _b1
h1 = h1.sigmoid()

_out = w2.mm(h1) + _b2
_out = _out.sigmoid()

loss = ((_out - _ty) * (_out - _ty) / 2).sum()  # here can use sum() or mean()
loss.backward()  # start backwarding
print(_out, loss)
print(w2.grad)
print(w1.grad)

w2.data.sub_(0.5 * w2.grad)
w1.data.sub_(0.5 * w1.grad)
h1 = w1.mm(_input) + _b1
h1 = h1.sigmoid()

_out = w2.mm(h1) + _b2
_out = _out.sigmoid()

loss = ((_out - _ty) * (_out - _ty) / 2).sum()  # here can use sum() or mean()
loss.backward()  # start backwarding
print(_out, loss)
