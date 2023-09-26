import torch
import torchvision
import torch.utils.data
from torchvision import transforms
from d2l import torch as d2l
d2l.use_svg_display()
trans= transforms.ToTensor()
mnist_train=torchvision.datasets.FashionMNIST(
    root="../data",train=True,transform=trans,download=True)
mnist_test=torchvision.datasets.FashionMNIST(
    root="../data",train=False,transform=trans,download=True)
print(len(mnist_train),len(mnist_test))
print(mnist_train[0][0].shape)
"""转化标签"""
def get_fashion_mnist_labels(labels):
    text_labels=["t-shirt","trouser","pullover","dress","coat","sandal","shirt","sneaker","bag","ankle boot"]
    return [text_labels[int(i)]for i in labels]
def show_image(imgs,num_rows,num_cols,titles=None,scale=1.5):
    figsize=(num_cols*scale,num_rows*scale)
    _,axes=d2l.plt.subplots(num_rows,num_cols,figsize=figsize)
    axes=axes.flatten()
    for i,(ax,img) in enumerate(zip(axes,imgs)):
        if torch.is_tensor(img):
            #张量图
            ax.imshow(img.numpy)
        else:
            #plt图
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

