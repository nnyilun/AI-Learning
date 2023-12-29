import torch
from torch.utils import data
from torch import nn
import torchvision
from torchvision import transforms
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
from IPython import display
# from alive_progress import alive_bar # alive-progress无法在jupyter正常显示
from tqdm import tqdm

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./fashionmnist/data/fashion/", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./fashionmnist/data/fashion/", train=False, transform=trans, download=True)
    
    # 将启用数据从 CPU 到 GPU 的固定内存复制
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, pin_memory=True),
            data.DataLoader(mnist_test, batch_size, shuffle=False, pin_memory=True))

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    figsize = (num_cols * scale, num_rows * scale) 
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize) 
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n
        
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
        
    def reset(self):
        self.data = [0.0] * len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]
    
    
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


batch_size = 1024
train_iter, test_iter = load_data_fashion_mnist(batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10)).to(device)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.0)
        
net.apply(init_weights)

loss = nn.CrossEntropyLoss(reduction='none').to(device)
optim = torch.optim.SGD(net.parameters(), lr=0.1)

def train(net, train_iter, loss, num_epochs, optim):
    for epoch in range(num_epochs):
        if isinstance(net, nn.Module):
            net.train()
        
        # 训练损失总和、训练准确度总和、样本数
        metric = Accumulator(3)
        
        progress_bar = tqdm(train_iter, desc='Processing', leave=True)
        for X, y in progress_bar:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_hat = net(X)
            l = loss(y_hat, y)
            optim.zero_grad()
            l.mean().backward()
            optim.step()
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())

        print(f'epoch:{epoch}, train_loss:{metric[0] / metric[2]}, train_acc:{metric[1] / metric[2] }')
    print(f'Done!Finished {num_epochs} epochs.')

print('start...')
num_epochs = 20
train(net, train_iter, loss, num_epochs, optim)