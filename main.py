from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#Operations on tensors, numpy to tensor, tensor to cuda
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
def tensoroperations():
    x = torch.rand(5, 3)
    print(x)
    x=torch.zeros(5,3,dtype=torch.long)
    print(x)
    x=torch.tensor([5.5,3])
    print(x)
    x=x.new_ones(5,3,dtype=torch.double)
    print(x)
    x= torch.randn_like(x,dtype=torch.float)
    print(x)
    print(x.size())
    y=torch.rand(5,3)
    print(x+y)
    print(torch.add(x,y))
    result = torch.empty(5,3)
    torch.add(x,y,out=result)
    print(result)
    y.add_(x)
    print(y)
    print(x[:,1])
    x=torch.randn(4,4)
    y=x.view(16)
    z=x.view(-1,8)
    print(x.size(),y.size(),z.size())
    x=torch.randn(1)
    print(x)
    print(x.item())
    a=torch.ones(5)
    print(a)
    b=a.numpy()
    print(b)
    a.add_(1)
    print(a)
    print(b)

    import numpy as np
    a = np.ones(5)
    b=torch.from_numpy(a)
    print(a)
    print(b)
    if torch.cuda.is_available():
        device=torch.device("cuda")
        y=torch.ones_like(x,device=device)
        x=x.to(device)
        z= x+y
        print(z)
        print(z.to("cpu",torch.double))
#Differentiations
def differentiations():
    x = torch.ones(2, 2, requires_grad=True)
    print(x)
    y=x+2
    print(y)
    print(y.grad_fn)
    z= y*y*3
    out = z.mean()
    print(z,out)
    a = torch.randn(2,2)
    a=((a*3)/(a-1))
    print(a.requires_grad)
    a.requires_grad_(True)
    print(a.requires_grad)
    b= (a*a).sum()
    print(b.grad_fn)
    out.backward()
    print(x.grad)
    x= torch.randn(3,requires_grad=True)
    y=x*2
    while y.data.norm() < 1000 :
        y = y*2
    print(y)
    v= torch.tensor([0.1,1.0,0.0001],dtype=torch.float)
    y.backward(v)
    print(x.grad)
    print(x.requires_grad)
    print((x**2).requires_grad)
    with torch.no_grad():
        print((x**2).requires_grad)
    print(x.requires_grad)
    y=x.detach()
    print(y.requires_grad)
    print(x.eq(y).all())
def neuralnetworkex():
    net = Net()
    print(net)
    params = list(net.parameters())
    print(len(params))
    print(params[0].size())  # conv1's .weight
    input = torch.randn(1, 1, 32, 32)
    out = net(input)
    print(out)
    net.zero_grad()
    out.backward(torch.randn(1, 10))
    output = net(input)
    target = torch.randn(10)  # a dummy target, for example
    target = target.view(1, -1)  # make it the same shape as output
    # create your optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    criterion = nn.MSELoss()




    loss = criterion(output, target)
    print(loss)
    print(loss.grad_fn)  # MSELoss
    print(loss.grad_fn.next_functions[0][0])  # Linear
    print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
    net.zero_grad()  # zeroes the gradient buffers of all parameters

    print('conv1.bias.grad before backward')
    print(net.conv1.bias.grad)

    loss.backward()

    print('conv1.bias.grad after backward')
    print(net.conv1.bias.grad)


    # in your training loop:
    optimizer.zero_grad()  # zero the gradient buffers
    output = net(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()  # Does the update

if __name__ == "__main__":
    neuralnetworkex()
