import torch
import torchvision
import torchvision.transforms as transforms

class Cifar10Loaders:
    """
    Data loaders for cifar 10 dataset
    """
    def __init__(self, data_dir='/tmp/cifar10', download=True, batch_size=128, pin_memory=True, num_workers=4):
        self.data_dir = data_dir
        self.download = download
        self.batch_size= batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
    def train_loader(self,shuffle=True):
        trainset = torchvision.datasets.CIFAR10(root=self.data_dir, train=True, download=True, transform=self.train_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers, pin_memory=self.pin_memory)
        return trainloader
    
    def test_loader(self,shuffle=False):
        testset = torchvision.datasets.CIFAR10(root=self.data_dir, train=False, download=True, transform=self.test_transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers, pin_memory=self.pin_memory)
        return testloader
    
    


