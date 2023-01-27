from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys
import random
from sklearn.model_selection import train_test_split
from torchvision import io as tio

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')   

class Caltech(VisionDataset):

    def retrieveTrainVal(self):
        samples = range(0,len(self.data))
        labels = []
        for i in samples:
            labels.append(self.data[i][1])
        train, val, y_train, y_val = train_test_split(  samples,labels,test_size=0.5,
                                                        random_state=42,stratify=labels)
        print("retrieveTrainVal")
        return train, val

    def __init__(self, root="/homes/vpipoli/VittorioPipoli", split='train', transform=None):
        super(Caltech, self).__init__(root, transform=transform)

        self.split = split  # This defines the split you are going to use
        self.data = []
        self.classes = []
        # (split files are called 'train.txt' and 'test.txt')

        provData = []

        f = open(f"{root}/Caltech101/{split}_cutted.txt")
        for x in f:
            dire = f"{root}/Caltech101/101_ObjectCategories/{x.rstrip()}"
            filename = x.split("/")
            if filename[0] != "BACKGROUND_Google":
                couple = []
                if filename[0] in self.classes:
                    img = pil_loader(dire)
                    index = len(self.classes) - 1
                    couple.append(img)
                    couple.append(index)
                    self.data.append(couple)
                else:
                    self.classes.append(filename[0])
                    img = pil_loader(dire)
                    index = len(self.classes) - 1
                    couple.append(img)
                    couple.append(index)
                    self.data.append(couple)

        f.close()
        print(len(self.data))
        print(len(self.classes))
        print(self.classes)


    def __getitem__(self, index):
        image, label = self.data[index][0], self.data[index][1]  
        if self.transform is not None:
            image = self.transform(image)

        if label > 0:
            label = 1
        return image, label

    def __len__(self):
        length = len(self.data)  # Provide a way to get the length (number of elements) of the dataset
        return length

########################################


# cal = Caltech()
# train, val = cal.retrieveTrainVal()
# image, label = cal.__getitem__(train[0])
# print(image, label)