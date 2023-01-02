from torchvision import datasets, transforms


def get_dataset(data_dir, training=True):
    trsfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    return datasets.FashionMNIST(data_dir, train=training, download=True, transform=trsfm)

