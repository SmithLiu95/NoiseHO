import torchvision.transforms as transforms
from data.cifar import CIFAR10, CIFAR100

def build_CIFAR(args,transform_train=None,valTrain=False):
    if transform_train is None:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
        ])
        
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
        ])    

    train_dataset = CIFAR10(root='./data/',
                                download=True,  
                                type='train', 
                                transform=transform_train,
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate,
                                small=True
                                )

    val_dataset = CIFAR10(root='./data/',
                                download=True,  
                                type='validation', 
                                transform=transform_train if valTrain else transform_test,
                                noise_type='clean',
                                noise_rate=0
                                )

    test_dataset = CIFAR10(root='./data/',
                                download=True,  
                                type='test', 
                                transform=transform_test,
                                noise_type='clean',
                                noise_rate=0
                                )
    return train_dataset,val_dataset,test_dataset