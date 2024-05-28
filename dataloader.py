from torchvision import datasets
from torchvision.transforms import transforms
import warnings
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=UserWarning)

train_data = datasets.MNIST(
    root = 'data',
    train = True,
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]),
    download = True
)

test_data = datasets.MNIST(
    root = 'data',
    train = False,
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]),
    download = True
)

loaders = {
    'train': DataLoader(train_data, batch_size=100, shuffle=True, num_workers=1),
    'test': DataLoader(test_data, batch_size=100, shuffle=True, num_workers=1)
}

def get_loader():
    return loaders