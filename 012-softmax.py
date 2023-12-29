import torch
from torch.utils import data
import torchvision
from torchvision import transforms 
from IPython import display
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
import fashionmnist.utils.mnist_reader as mnist_reader


mnist_train_img, mnist_train_label = mnist_reader.load_mnist('./fashionmnist/data/fashion/', kind='train')
mnist_test_img, mnist_test_label = mnist_reader.load_mnist('./fashionmnist/data/fashion/', kind='t10k')
mnist_train_img, mnist_train_label = torch.tensor(mnist_train_img, dtype=torch.float32).reshape(-1, 1, 28, 28), torch.tensor(mnist_train_label, dtype=torch.int64)
mnist_test_img, mnist_test_label = torch.tensor(mnist_test_img, dtype=torch.float32).reshape(-1, 1, 28, 28), torch.tensor(mnist_test_label, dtype=torch.int64)


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


class MnistDataset(data.Dataset):
    def __init__(self, inputs, labels, transform=None):
        self.inputs = inputs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input_sample = self.inputs[index]
        label_sample = self.labels[index]
        
        if self.transform:
            input_sample = self.transform(input_sample)
        
        return input_sample, label_sample
    

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


def get_dataloader_workers():  #@save
    """使用4个进程来读取数据"""
    return 4  # 并行

def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
        #transforms.Resize：调整PILImage对象的尺寸。transforms.Resize([h, w])或transforms.Resize(x)等比例缩放
    trans = transforms.Compose(trans) # 串联多个图片变换的操作
    mnist_train = torchvision.datasets.FashionMNIST(
        root="C:/Users/jyl/Desktop/jupyter/1", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="C:/Users/jyl/Desktop/jupyter/1", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))


# def load_data_fashion_mnist(batch_size, resize=None):
#     trans = []
#     if resize:
#         trans = [
#                   transforms.ToPILImage(), 
#                   transforms.Resize(resize), 
#                   transforms.ToTensor(),
#                 ]
#     trans = transforms.Compose(trans)
    
#     train_dateset = MnistDataset(mnist_train_img, mnist_train_label, transform=trans)
#     test_dataset = MnistDataset(mnist_test_img, mnist_test_label, transform=trans)
    
#     return (data.DataLoader(train_dateset, batch_size, shuffle=True, num_workers=0),
#             data.DataLoader(test_dataset, batch_size, shuffle=False, num_workers=0))


def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition


def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


def find_nan_indices(tensor):
    nan_indices = torch.isnan(tensor)
    nan_indices = torch.nonzero(nan_indices)
    return nan_indices


def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n
        
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
        
    def reset(self):
        self.data = [0.0] * len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]



def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    
    metric = Accumulator(3)
    
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


class Animator: #@save
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                ylim=None, xscale='linear', yscale='linear',
                fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        
        backend_inline.set_matplotlib_formats('svg')
        
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
    
    
    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                    legend=['train loss', 'train acc', 'test acc'])
  
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
        
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_() 


lr = 0.1

def updater(batch_size):
    return sgd([W, b], lr, batch_size)


batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)


num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)