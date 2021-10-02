from torch.utils.data import Dataset
from torch import Tensor

import cv2

import os

from typing import Union, List, Callable

class MonetDataset(Dataset):
    # def __init__(self, transforms: Union[None, List[Callable]] = None):
    def __init__(self):
        super()
        self.files_path = "./data/datasets/gan-getting-started/monet_jpg"
        self.file_names = os.listdir(self.files_path)

    
    def __len__(self) -> int:
        return len(self.file_names)

    
    def __getitem__(self, index) -> Tensor:
        img_path = os.path.join(self.files_path, self.file_names[index])
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return Tensor(img)

    
class PhotoDataset(Dataset):
    def __init__(self):
        super()
        self.files_path = "./data/datasets/gan-getting-started/photo_jpg"
        self.file_names = os.listdir(self.files_path)

    
    def __len__(self) -> int:
        return len(self.file_names)

    
    def __getitem__(self, index) -> Tensor:
        img_path = os.path.join(self.files_path, self.file_names[index])
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return Tensor(img)


class MonetLoader()

    