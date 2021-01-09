```
python main.py --ho --n_epoch 0 --n_epoch_hyper 1  --hyper_scale=1000.0 --checkpoint='cifar10_0.5symmetric_resnet18_trainLoss0.03.pt'
```

ho: enable hyperparameter optimization

General idea:
```
for i in [0,n_epoch_hyper]:
    train network for e_epoch with normal training process
    update hyperparameters
```

