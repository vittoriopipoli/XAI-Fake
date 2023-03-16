from torchvision import transforms
from torch import nn
import numpy as np
import torch

class SpectrumShift(nn.Module):

    def __init__(self, fft_img_real_mean, fft_img_fake_mean, fake_to_real=None):
        super().__init__()
        self.fft_img_real_mean = torch.squeeze(torch.tensor(fft_img_real_mean), 0)
        self.fft_img_fake_mean = torch.squeeze(torch.tensor(fft_img_fake_mean), 0)
        self.fake_to_real      = fake_to_real
        self.diff              = self.fft_img_real_mean-self.fft_img_fake_mean


    def forward(self, img):        
        fft = torch.fft.fft2(img)
        img_real = torch.real(fft)
        img_imag = torch.imag(fft)
        img_pha  = torch.atan2(img_imag, img_real)
        img_mod  = torch.fft.fftshift(torch.sqrt(torch.pow(img_real, 2)+torch.pow(img_imag, 2)))
        if self.fake_to_real == True:
            img_mod = img_mod + self.diff
        elif self.fake_to_real == False:
            img_mod = img_mod - self.diff
        else:
            pass
        img_mod = torch.fft.ifftshift(img_mod)
        inverse = torch.real(torch.fft.ifft2(
                        torch.multiply(img_mod.type(torch.complex64), torch.exp(torch.tensor(1j, dtype=torch.complex64)*img_pha.type(torch.complex64)))
                        ))        
        return inverse


    def __repr__(self):
        return self.__class__.__name__
    
if __name__=="__main__":
    BATCH_SIZE, C, H, W = 8, 1, 224, 224
    img = torch.rand(BATCH_SIZE, C, H, W)
    fft_img_real_mean = torch.rand(C, H, W)
    fft_img_fake_mean = torch.rand(C, H, W)
    ss = SpectrumShift(fft_img_real_mean, fft_img_fake_mean, True)
    res = ss(img)
    print(res)
    