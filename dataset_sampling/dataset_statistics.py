import numpy
import pandas as pd
import torch
from data.fake_dataset_dloader import FakeDataset
import torchvision.transforms as transforms
from scipy.fftpack import dct, idct
from skimage.io import imread
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pylab as plt

def compute_stats(annotation_file):
    transform = transforms.Compose([transforms.RandomEqualize(p=1),transforms.ToTensor()])
    fake_dataset = FakeDataset(annotation_file, transform)
    psum_real = torch.tensor([0.0, 0.0, 0.0])
    psum_sq_real = torch.tensor([0.0, 0.0, 0.0])
    psum_fake = torch.tensor([0.0, 0.0, 0.0])
    psum_sq_fake = torch.tensor([0.0, 0.0, 0.0])

    len_real = 0
    len_fake = 0
    # loop through images
    for inputs, label in fake_dataset:
        if label == 0:
            psum_real += inputs.sum(axis=[1, 2])
            psum_sq_real += (inputs ** 2).sum(axis=[1, 2])
            len_real += inputs.shape[1] * inputs.shape[2]
        else:
            psum_fake += inputs.sum(axis=[1, 2])
            psum_sq_fake += (inputs ** 2).sum(axis=[1, 2])
            len_fake += inputs.shape[1] * inputs.shape[2]

    # mean and std
    total_mean_real = psum_real / len_real
    total_var_real = (psum_sq_real / len_real) - (total_mean_real ** 2)
    total_std_real = torch.sqrt(total_var_real)

    # output
    print('mean_real: ' + str(total_mean_real))
    print('std_real:  ' + str(total_std_real))

    # mean and std
    total_mean_fake = psum_fake / len_fake
    total_var_fake = (psum_sq_fake / len_fake) - (total_mean_fake ** 2)
    total_std_fake = torch.sqrt(total_var_fake)

    # output
    print('mean_fake: ' + str(total_mean_fake))
    print('std_fake:  ' + str(total_std_fake))

# implement 2D DCT
def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

# implement 2D IDCT
def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')


def compute_spectr(annotation_file):
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    fake_dataset = FakeDataset(annotation_file, transform)
    fft_img_real = numpy.zeros((224,224))
    fft_img_fake = numpy.zeros((224,224))
    real_count = 0
    fake_count = 0
    for element, target, _ in fake_dataset:
        r,g,b = element.unbind(0)
        if target == 0:
            fft_img_real = fft_img_real + numpy.abs((torch.fft.fftshift(torch.fft.fft2(r))).numpy())
            fft_img_real = fft_img_real + numpy.abs((torch.fft.fftshift(torch.fft.fft2(g))).numpy())
            fft_img_real = fft_img_real + numpy.abs((torch.fft.fftshift(torch.fft.fft2(b))).numpy())
            real_count +=1
        else:
            fft_img_fake = fft_img_fake + numpy.abs((torch.fft.fftshift(torch.fft.fft2(r))).numpy())
            fft_img_fake = fft_img_fake + numpy.abs((torch.fft.fftshift(torch.fft.fft2(g))).numpy())
            fft_img_fake = fft_img_fake + numpy.abs((torch.fft.fftshift(torch.fft.fft2(b))).numpy())
            fake_count += 1
    fft_img_real = fft_img_real / (real_count * 3)
    fft_img_fake = fft_img_fake / (fake_count * 3)
    plt.imshow(np.log(np.abs(fft_img_real))), plt.axis('off'), plt.title("Real"), plt.savefig('Real'), plt.show(), plt.close()
    plt.imshow(np.log(np.abs(fft_img_fake))), plt.axis('off'), plt.title("Fake"), plt.savefig('Fake'), plt.show(), plt.close()

#imF = imF + dct2(r)
        # im1 = idct2(imF)
        #
        # # check if the reconstructed image is nearly equal to the original image
        # np.allclose(r, im1)
        # # True
        #
        # # plot original and reconstructed images with matplotlib.pylab
        # #plt.gray()
        #
        # #plt.subplot(121), plt.imshow(im), plt.axis('off')
        # #plt.subplot(122), plt.imshow(im1), plt.axis('off')
        #plt.imshow(fft_img.real.numpy()), plt.axis('off')

if __name__ == "__main__":
    #compute_stats("dataset_10000.csv")
    compute_spectr("dataset_10000.csv")
