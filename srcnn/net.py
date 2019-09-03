"""
Copyright (C) Microsoft Corporation. All rights reserved.​
 ​
Microsoft Corporation ("Microsoft") grants you a nonexclusive, perpetual,
royalty-free right to use, copy, and modify the software code provided by us
("Software Code"). You may not sublicense the Software Code or any use of it
(except to your affiliates and to vendors to perform work on your behalf)
through distribution, network access, service agreement, lease, rental, or
otherwise. This license does not purport to express any claim of ownership over
data you may have shared with Microsoft in the creation of the Software Code.
Unless applicable law gives you more rights, Microsoft reserves all other
rights not expressly granted herein, whether by implication, estoppel or
otherwise. ​
 ​
THE SOFTWARE CODE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
MICROSOFT OR ITS LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THE SOFTWARE CODE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

configs = [()]


def make_layers(Bn=True, input=256):
    global configs
    layers = []
    layer = nn.Conv2d(input, input, kernel_size=1, stride=1, padding=0)
    layers.append(layer)
    if Bn:
        layers.append(nn.BatchNorm2d(input))

    for k, s, c in configs:
        if c == -1:
            layer = nn.Conv2d(kernel_size=k, stride=s, padding=0)
        else:
            now = []
            now.append(nn.Conv1d(input, c, kernel_size=k, stride=s, padding=0))
            input = c
            if Bn:
                now.append(nn.BatchNorm2d(input))
            now.append(nn.Relu(inplace=True))
            layer = nn.Sequential(*now)
        layers.append(layer)
    return nn.Sequential(*layers), input


class trynet(nn.Module):
    def __init__(self):
        super(trynet, self).__init__()
        self.layer1 = nn.Conv1d(1, 128, kernel_size=128, stride=0, padding=0)
        self.layer2 = nn.BatchNorm1d(128)

        self.feature = make_layers()


class Anomaly(nn.Module):
    def __init__(self, window=1024):
        self.window = window
        super(Anomaly, self).__init__()
        self.layer1 = nn.Conv1d(window, window, kernel_size=1, stride=1, padding=0)
        self.layer2 = nn.Conv1d(window, 2 * window, kernel_size=1, stride=1, padding=0)
        self.fc1 = nn.Linear(2 * window, 4 * window)
        self.fc2 = nn.Linear(4 * window, window)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(x.size(0), self.window, 1)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)


def save_model(model, model_path):
    try:
        torch.save(model.state_dict(), model_path)
    except:
        torch.save(model, model_path)


def load_model(model, path):
    print("loading %s" % path)
    with open(path, 'rb') as f:
        pretrained = torch.load(f, map_location=lambda storage, loc: storage)
        model_dict = model.state_dict()
        pretrained = {k: v for k, v in pretrained.items() if k in model_dict}
        model_dict.update(pretrained)
        model.load_state_dict(model_dict)
    return model
