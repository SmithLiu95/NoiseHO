import torch
import torch.nn.functional as F
def model_eval(model,loader,p=False):
    N_data = len(loader.dataset)
    loss, corr = 0, 0
    with torch.no_grad():
        for data, target,_ in loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss += F.cross_entropy(output,target,reduction='sum')
            pred = output.argmax(dim=1, keepdim=True)
            corr += pred.eq(target.view_as(pred)).sum().item()
    loss /= N_data
    corr *= (100.0/N_data)
    return loss, corr


# evaludate loss/acc
def model_test(epoch_ho,model,train_loader,val_loader,test_loader,lr):
    model.eval()
    
    train_loss, train_corr = model_eval(model,train_loader)
    val_loss, val_corr = model_eval(model,val_loader)
    test_loss, test_corr = model_eval(model,test_loader)
    
    print('Epoch {} Loss: {:.4f}, {:.4f}, {:.4f}, Accuracy: {:.2f}% {:.2f}% {:.2f}% lr: {:.6f}'.format(
            epoch_ho, train_loss, val_loss, test_loss, train_corr, val_corr, test_corr,lr))
    return val_corr

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

import torchvision.transforms as transforms
def dataset_switch_mode(dataloader,train=True):
    if not train:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
        ])  
    else:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
        ]) 
    dataloader.dataset.transform=transform
    return dataloader
    

from torch import nn
class WeightedCE(nn.Module):
    def __init__(self,num_data,num_class,loss=nn.CrossEntropyLoss):
        super(WeightedCE,self).__init__()
        self.loss=loss(reduction='none')
        self.weight_instance=nn.Parameter(torch.rand(num_data).float())
    def get_weight(self):
        weight_instance=self.weight_instance/torch.max(self.weight_instance)
        weight_instance=F.relu(weight_instance)
        return weight_instance
        
    def forward(self, y_pred,y_true,idx):
        if idx is not None:
            weight_instance=self.get_weight()[idx]
            output = weight_instance*self.loss(y_pred,y_true)
            output = torch.sum(output)/torch.sum(weight_instance)
        else:
            output = self.loss(y_pred,y_true)
            output = torch.mean(output)
        return output