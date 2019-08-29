import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
configs =  [()]
def make_layers(Bn=True,input=256):
    global configs
    layers = []
    layer = nn.Conv2d(input,input,kernel_size=1,stride=1,padding=0)
    layers.append(layer)
    if Bn:
        layers.append(nn.BatchNorm2d(input))

    for k,s,c in configs:
        if c == -1:
            layer = nn.Conv2d(kernel_size=k,stride=s,padding=0)
        else:
            now = []
            now.append(nn.Conv1d(input,c,kernel_size=k,stride=s,padding=0))
            input=c
            if Bn:
                now.append(nn.BatchNorm2d(input))
            now.append(nn.Relu(inplace = True))
            layer = nn.Sequential(*now)
        layers.append(layer)
    return nn.Sequential(*layers),input
class trynet(nn.Module):
    def __init__(self):
        super(trynet, self).__init__()
        self.layer1 = nn.Conv1d(1,128,kernel_size=128,stride=0,padding=0)
        self.layer2 = nn.BatchNorm1d(128)
        
        self.feature = make_layers()


class Anomaly(nn.Module):
    def __init__(self,window=1024):
        self.window=window
        super(Anomaly, self).__init__()
        # self.layer1 = nn.Conv1d(1,256,kernel_size=128,stride=1,padding=0)
        self.layer1 = nn.Conv1d(window,window,kernel_size=1,stride=1,padding=0)
        self.layer2 = nn.Conv1d(window,2*window,kernel_size=1,stride=1,padding=0)
        self.fc1 = nn.Linear(2*window,4*window)
        self.fc2 = nn.Linear(4*window,window)
        self.relu = nn.ReLU(inplace = True)
        

    def forward(self, x):
        # print('before : ',x.size())
        x = x.view(x.size(0),self.window,1)
        # print('after :',x.size())
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = x.view(x.size(0),-1)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # z = self.reparameterize(mu, logvar)
        return torch.sigmoid(x)
        # return x


def save_model(model,model_path):

    try:
        torch.save(model.state_dict(), model_path)
    except:
        torch.save(model,model_path)


def load_model(model,path):
    print("loading %s" % path)
    with open(path,'rb') as f:
        pretrained = torch.load(f, map_location=lambda storage, loc: storage)
        model_dict = model.state_dict()
        pretrained = {k:v for k,v in pretrained.items() if k in model_dict}
        model_dict.update(pretrained)
        model.load_state_dict(model_dict)
    return model
