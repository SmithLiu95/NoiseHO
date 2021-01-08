import argparse
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

import torch,tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from warmup_scheduler import GradualWarmupScheduler
import torchvision.transforms as transforms

from mydataset import build_CIFAR


def main(args,model):

    exp_name='HOv5_{}_{}{}'.format(args.dataset,args.noise_rate,args.noise_type)
    if args.valTrain:
        exp_name+='_VAL'
    exp_name+='_{}_epoch{}x{}_lr{}'.format(args.model,args.n_epoch_hyper,args.n_epoch,args.lr)
    if args.ho:
        exp_name+='_i{}_hyperLR{}_maxGradNorm{}'.format(args.i_neumann,args.lr_hyper,args.max_grad_norm)
    if args.cosineLRdecay:
        exp_name+='_cosineLRdecay'
    exp_name+='_id{}'.format(args.exp_id)
    print(exp_name)
    # exit()
    # Seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Hyper Parameters
    batch_size = 64
    learning_rate = args.lr 
        
    if args.dataset=='cifar10':
        input_channel=3
        init_epoch = 20
        num_classes = 10

        train_dataset,val_dataset,test_dataset=build_CIFAR(args,valTrain=args.valTrain)


    if args.forget_rate is None:
        forget_rate=args.noise_rate
    else:
        forget_rate=args.forget_rate

    if args.valTrain:
        train_dataset,val_dataset=val_dataset,train_dataset #swap dataset
    else:
        noise_or_not = train_dataset.noise_or_not
    print('loading dataset...')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=batch_size, 
                                                num_workers=args.num_workers,
                                                drop_last=False,
                                                shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                batch_size=batch_size, 
                                                num_workers=args.num_workers,
                                                drop_last=False,
                                                shuffle=False)
        
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=batch_size, 
                                                num_workers=args.num_workers,
                                                drop_last=False,
                                                shuffle=False)
    print('train',train_dataset.type,train_dataset.noise_type,len(train_dataset))                
    print('val',val_dataset.type,val_dataset.noise_type,len(val_dataset))       
    print('test',test_dataset.type,test_dataset.noise_type,len(test_dataset))    



    criateria = WeightedCE(len(train_dataset.train_data),num_classes).cuda()

    from cyy_naive_pytorch_lib.model_loss import ModelWithLoss
    model_with_loss = ModelWithLoss(model=model,loss_fun=criateria)
    
    # init state
    num_epochs = args.n_epoch
    num_epochs_hyper = args.n_epoch_hyper

    # classification model
    # optimizer for classification model
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,weight_decay=args.weight_decay,momentum=0.9)

    # scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max',factor=0.1,patience=5)
    if not args.cosineLRdecay:
        scheduler_plateau = torch.optim.lr_scheduler.StepLR(optimizer,150,0.1)
    else:
        scheduler_plateau = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.n_epoch*args.n_epoch_hyper)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.warmup, after_scheduler=None)

    # criateria = nn.CrossEntropyLoss()



    if args.ho:
        # hyperparamter model
        # optimizer for hyperparameter model
        optimizer_ho = optim.SGD(criateria.parameters(), lr=args.lr_hyper)
    
    import time
    t0=time.time()
    # model_test(0)
    # hyperparameter tuning
    best_val_corr=0
    for epoch_ho in range(num_epochs_hyper):
        # optimize classification model with current hyperparameters
        for epoch in range(num_epochs):
            if epoch_ho*num_epochs+epoch<=args.warmup and args.checkpoint=='none':
                scheduler_warmup.step()
            model.train()
            for batch_idx, (data, target,indices) in enumerate(train_loader):
                data, target = data.cuda(), target.cuda()
                optimizer.zero_grad()
                y=model(data)
                loss = criateria(y,target,indices)
                loss.backward()
                optimizer.step()
            val_corr=model_test(epoch,model,train_loader,val_loader,test_loader,get_lr(optimizer))
            if epoch_ho*num_epochs+epoch>args.warmup:
                scheduler_plateau.step()
            if val_corr>best_val_corr:
                torch.save(model.state_dict(), '{}_best.pt'.format(exp_name))
                best_val_corr=val_corr
            torch.save(model.state_dict(), '{}_last.pt'.format(exp_name))


        
        if args.ho:
            model.eval()

            # calculate hypergradient
            # for validation
            model.zero_grad()
            criateria.zero_grad()
            optimizer.zero_grad()

            for batch_idx, (data, target,_) in enumerate(val_loader):
                data, target = data.cuda(), target.cuda()

                y=model(data)
                loss_V = criateria(y,target,None)
                break
            loss_V.backward()

            v1=[p.grad.view(-1).clone() for n, p in model.named_parameters()]
            v1=torch.cat(v1)
                
                
            # for calculating training loss
            model.zero_grad()
            criateria.zero_grad()
            optimizer.zero_grad()

            for batch_idx, (data, target,indices) in enumerate(train_loader):
                data, target = data.cuda(), target.cuda()

                y=model(data)
                loss_T = criateria(y, target,indices)
                break
            
            w_T=[p for n, p in model.named_parameters()]
            f = torch.autograd.grad(loss_T, w_T, create_graph=True) # dLt/dw

            from inverse_hessian_vector_product import stochastic_inverse_hessian_vector_product
            v2=stochastic_inverse_hessian_vector_product(dataset=train_dataset,model_with_loss=model_with_loss,v=v1,batch_size=batch_size,scale=args.hyper_scale)


        
            lmd = list(criateria.parameters())
            
            v3 = torch.autograd.grad(f, lmd, grad_outputs=v2,allow_unused=True)[0] # d2Lt/dwdl.

    
            # update hyperparameters
            hyper_grad = 0 - v3
            optimizer_ho.zero_grad()
            criateria.weight_instance.grad = hyper_grad
            # print(v3.norm(2))
            torch.nn.utils.clip_grad_norm_(criateria.parameters(),args.max_grad_norm)
            optimizer_ho.step()

            with torch.no_grad():
                weight=criateria.get_weight()
                print('mean: {:.4f} L2norm: {:.4f} var: {:.4f}'.format(torch.mean(weight),criateria.weight_instance.norm(2),torch.var(weight)))

        print('epch_ho {} done,best val accuracy={:.2f}'.format(epoch_ho,best_val_corr))

    print('total training time: {:.4f}'.format(time.time()-t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type = float, default = 0.02)
    parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.5)
    parser.add_argument('--forget_rate', type = float, help = 'forget rate', default = None)
    parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric]', default='symmetric')
    parser.add_argument('--dataset', type = str, help = 'mnist, cifar10, cifar100, or imagenet_tiny', default = 'cifar10')
    parser.add_argument('--n_epoch', type=int, default=300)
    parser.add_argument('--n_epoch_hyper', type=int, default=1)
    parser.add_argument('--model', type = str, default='resnet18')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=8, help='how many subprocesses to use for data loading')
    parser.add_argument('--num_batch_hyper', type=int, default=1)
    parser.add_argument('--lr_hyper', type=float, default=0.1)
    parser.add_argument('--i_neumann', type=int, default=20)
    parser.add_argument('--exp_id', type=int, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=50)
    parser.add_argument('--ho', action='store_true')
    parser.add_argument('--valTrain', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--checkpoint', type = str, default='none')
    parser.add_argument('--hyper_scale', type=float, default=1000.0)
    parser.add_argument('--cosineLRdecay', action='store_true')
    args = parser.parse_args()
    assert (args.valTrain and args.ho)==False

    num_classes=10
    from utils import model_test,get_lr,WeightedCE
    from torchvision.models.resnet import resnet18
    model = resnet18(num_classes=num_classes).cuda()
    if args.checkpoint!='none':
        model.load_state_dict(torch.load(args.checkpoint))
    main(args,model,)
