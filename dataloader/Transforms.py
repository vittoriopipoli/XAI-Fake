# ########################
# FUNCTION THAT RETURNS A LIST OF ALL THE 
# PREPROCESSING FUNCTIONS ACCORDING TO THE PREPROCESSING.YAML FILE

# #########################
# FUNCTION THAT RETURNS A LIST OF ALL THE 
# AUGMENTATION FUNCTIONS ACCORDING TO THE AUGMENTATION.YAML FILE

# ###########################
# FUNCTION THAT CONCATENATES THE LISTS OF THE
# PREPROCESSING AND THE AUGMENTATIONS FUNCTIONS (IF ANY)
# WITH THE NORMALIZE FUNCTION AND RETURNS
# THE TRANSFORM COMPOSE OF THE RESULTING LIST
from torchvision import transforms
from torch import nn

def create_transforms(pre_proccessing, augmentation, config, eval=False, compose=True):
    transform = []
    # if pre_proccessing.Equalize:
    #     transform.append(transforms.RandomEqualize(p=1))
    # if pre_proccessing.GaussianBlur:
    #     transform.append(transforms.GaussianBlur(5, sigma=(pre_proccessing.GaussianBlur.sigma)))
    if "Resize" in pre_proccessing:
        transform.append(transforms.Resize(pre_proccessing.Resize.size, interpolation=transforms.InterpolationMode.BICUBIC))
    if "CenterCrop" in pre_proccessing:
        transform.append(transforms.CenterCrop(pre_proccessing.CenterCrop.size))    
    if "Grayscale" in pre_proccessing:
        transform.append(transforms.Grayscale(num_output_channels=pre_proccessing.Grayscale.num_output_channels))        
    if "RandomHorizontalFlip" in pre_proccessing and not eval:
        transform.append(transforms.RandomHorizontalFlip(p=augmentation.RandomHorizontalFlip.p))
    if "RandomGrayscale" in pre_proccessing and not eval:
        transform.append(transforms.RandomGrayscale(p=augmentation.RandomGrayscale.p))
    if "RandomVerticalFlip" in pre_proccessing and not eval:
        transform.append(transforms.RandomVerticalFlip(p=augmentation.RandomVerticalFlip.p))

    transform.append(transforms.ToTensor())

    if "Normalize" in pre_proccessing:
        transform.append(transforms.Normalize(pre_proccessing.Normalization.mean, pre_proccessing.Normalization.std))
    if compose == True:
        return transforms.Compose(transform)
    else:
        return transform
