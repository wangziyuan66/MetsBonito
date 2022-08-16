from torch.utils.data import DataLoader
from torch.utils import data
class CustomDataset(data.Dataset):#需要继承data.Dataset
    def __init__(self,x,tf):
        # TODO
        # 1. Initialize file path or list of file names.
        self.X=x 
        self.y=tf
        self.length=len(self.X)
        pass
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        return self.X[index],self.y[index]
        
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return self.length