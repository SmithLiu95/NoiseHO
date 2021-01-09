```
python main.py --ho --n_epoch 0 --n_epoch_hyper 1  --hyper_scale=1000.0 --checkpoint='cifar10_0.5symmetric_resnet18_trainLoss0.03.pt'
```

ho: enable hyperparameter optimization

checkpoint is not a must. It is provided to make problem reproducing easy. The provided checkpoint is trained with no hyperparameter optimization, and achieved train loss L_{train} = 0.03

General idea:
```
for i in [0,n_epoch_hyper]:
    train network for n_epoch with normal training process
    update hyperparameters
```

