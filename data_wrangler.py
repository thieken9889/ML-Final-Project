import numpy as np
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def data_wrangler(datadir,imsize,show=False,batch_size=10):
    # create transforms for data
    # TODO we need to normalize the dataset!!!
    data_transform = transforms.Compose([transforms.Resize((imsize, imsize)), transforms.ToTensor()])

    # create the data set and data loader for train and test data
    train_ds = ImageFolder(root=(datadir+'\\train'), transform=data_transform)
    train_dl = DataLoader(train_ds,batch_size=batch_size,shuffle=True)

    test_ds = ImageFolder(root=(datadir+'\\test'), transform=data_transform)
    test_dl = DataLoader(test_ds,batch_size=batch_size,shuffle=True)

    if show:
        Xtr, _ = next(iter(train_dl))
        imshow(make_grid(Xtr[0:7, :, :, :]))

    return train_dl, test_dl


# ### Code from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html #####
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0)) # rearrange dimensions from (color,y,x) -> (y,x,color)
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # inp = std * inp + mean # undo the normalization
    inp = np.clip(inp, 0, 1)
    # Display image, without ticks
    plt.imshow(inp)
    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.show()  # added this to make the plot show in pycharm

###########################################################################################

