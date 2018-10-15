"""
For testing adaptation, we can try on some commonly-used image datasets
"""
import os
import numpy as np
import tensorflow as tf
import scipy.io as sio
import matplotlib.pyplot as plt
from mnist import MNIST # pip install --user python-mnist
from urllib.parse import urlparse
from urllib.request import urlretrieve

# Download and load the dataset.
def download(url, fn, force=False):
    """ Download url to a file called fn if it doesn't exist """
    if force or not os.path.exists(fn):
        urlretrieve(url, fn)
        print("Downloaded", fn)
    else:
        print("Already downloaded", fn)

def svhn(directory='datasets/svhn'):
    """ Load SVHN dataset, download if not already downloaded """
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Download
    files = [
        "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
        "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"
    ]

    for f in files:
        url_basename = os.path.basename(urlparse(f).path)
        download(f, os.path.join(directory, url_basename))

    # Load
    train = sio.loadmat(os.path.join(directory, 'train_32x32.mat'))
    test = sio.loadmat(os.path.join(directory, 'test_32x32.mat'))

    # Shape: [number of images, 32, 32, 3]
    train_images = np.transpose(train['X'], (3,0,1,2))
    test_images = np.transpose(test['X'], (3,0,1,2))
    # Shape: (number of images,), make it 0-9 rather than 1-10
    train_labels = np.squeeze(train['y'])
    test_labels = np.squeeze(test['y'])

    # Replace label 10 with 0 (since they made digit '0' be class '10')
    train_labels[train_labels == 10] = 0
    test_labels[test_labels == 10] = 0

    assert train_images.shape[0] == train_labels.shape[0]
    assert test_images.shape[0] == test_labels.shape[0]

    # Normalize
    train_images = (train_images.astype(np.float32) - 128.0) / 128.0
    test_images = (test_images.astype(np.float32) - 128.0) / 128.0

    return train_images, train_labels, test_images, test_labels

def mnist(directory='datasets/mnist'):
    """ Load MNIST dataset, download if not already downloaded """
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Download
    files = [
        "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
    ]

    for f in files:
        url_basename = os.path.basename(urlparse(f).path)
        download(f, os.path.join(directory, url_basename))

    # Load
    mndata = MNIST(directory, return_type='numpy', gz=True)
    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()
    assert train_images.shape[0] == train_labels.shape[0]
    assert test_images.shape[0] == test_labels.shape[0]

    # Reshape the 784x1 images to 28x28 images and normalize.
    train_images = (train_images.reshape(-1,28,28,1).astype(np.float32) - 128.0) / 128.0
    test_images = (test_images.reshape(-1,28,28,1).astype(np.float32) - 128.0) / 128.0

    # Pad image axes with zeros to be 32x32 to match SVHN
    # Note: since this is after normalization, black is now -1 not 0
    train_images = np.pad(train_images, ((0,),(2,),(2,),(0,)), 'constant', constant_values=-1)
    test_images = np.pad(test_images, ((0,),(2,),(2,),(0,)), 'constant', constant_values=-1)

    # Make this 3-channel like the SVHN
    train_images = np.tile(train_images, (1,1,1,3))
    test_images = np.tile(test_images, (1,1,1,3))

    return train_images, train_labels, test_images, test_labels

def show(images, labels=None, num=100, cols=10, title=None):
    """ View images """
    fig = plt.figure(figsize=(15,15))
    plt.axis('off')
    if title is not None:
        plt.suptitle(title, fontsize=14)
    fig.subplots_adjust(wspace=0, hspace=0.5)
    for i in range(num):
        ax = fig.add_subplot(np.ceil(num/cols), cols, i+1)
        ax.grid(False); ax.set_yticks([]); ax.set_xticks([])
        if labels is not None:
            ax.set_title(labels[i])
        plt.imshow((128*np.squeeze(images[i]) + 128).astype(np.uint8), cmap='gray')
    plt.show()

def denormalize(image):
    """ Take TensorFlow tensor and get a 8-bit (0-255) image again """
    return tf.cast(128*image + 128, tf.uint8)

if __name__ == "__main__":
    mi1, ml1, mi2, ml2 = mnist()
    si1, sl1, si2, sl2 = svhn()
    show(mi1, ml1, title="MNIST train")
    show(mi2, ml2, title="MNIST test")
    show(si1, sl1, title="SVHN train")
    show(si2, sl2, title="SVHN train")