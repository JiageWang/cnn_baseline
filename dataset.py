import pretrainedmodels
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


def get_data_loader(args):
    mean = pretrainedmodels.pretrained_settings[args.arch]['imagenet']['mean']
    std = pretrainedmodels.pretrained_settings[args.arch]['imagenet']['std']
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    valid_trainsform = transforms.Compose([
        transforms.Resize(int(args.input_size * 256 / 224)),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_datasets = ImageFolder(args.train_path, transform=train_transform)
    train_loader = DataLoader(train_datasets, batch_size=args.batch_size, shuffle=True)
    valid_datasets = ImageFolder(args.valid_path, transform=valid_trainsform)
    valid_loader = DataLoader(valid_datasets, batch_size=args.batch_size, shuffle=True)
    print("train set image number: ", len(train_datasets))
    print("valid set image number: ", len(valid_datasets))
    print("train set batch number: ", len(train_loader))
    print("valid set batch number: ", len(valid_loader))
    return train_loader, valid_loader

