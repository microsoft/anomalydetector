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

import pickle
import csv
import numpy as np
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.data import Dataset
from srcnn.net import *
import json
from msanomalydetector.util import average_filter
from msanomalydetector.spectral_residual import SpectralResidual


def read_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def read_csv_kpi(path):
    tm = []
    vl = []
    lb = []
    with open(path) as f:
        input = csv.reader(f, delimiter=',')
        cnt = 0
        for row in input:
            if cnt == 0:
                cnt += 1
                continue
            tm.append(int(row[0]))
            vl.append(float(row[1]))
            lb.append(int(row[2]))
            cnt += 1
        f.close()
    return tm, vl, lb


def read_csv(path):
    tm = []
    vl = []
    with open(path, 'r+') as f:
        input = csv.reader(f, delimiter=',')
        cnt = 0
        for row in input:
            if cnt == 0:
                cnt += 1
                continue
            tm.append(cnt)
            vl.append(float(row[1]))
        f.close()
    return tm, vl


def sr_cnn(data_path, model_path, win_size, lr, epochs, batch, num_worker, load_path=None):
    def adjust_lr(optimizer, epoch):
        base_lr = lr
        cur_lr = base_lr * (0.5 ** ((epoch + 10) // 10))
        for param in optimizer.param_groups:
            param['lr'] = cur_lr

    def Var(x):
        return Variable(x.cuda())

    def loss_function(x, lb):
        l2_reg = 0.
        l2_weight = 0.
        for W in net.parameters():
            l2_reg = l2_reg + W.norm(2)
        kpiweight = torch.ones(lb.shape)
        kpiweight[lb == 1] = win_size // 100
        kpiweight = kpiweight.cuda()
        BCE = F.binary_cross_entropy(x, lb, weight=kpiweight, reduction='sum')
        return l2_reg * l2_weight + BCE

    def calc(pred, true):
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for pre, gt in zip(pred, true):
            if gt == 1:
                if pre == 1:
                    TP += 1
                else:
                    FN += 1
            if gt == 0:
                if pre == 1:
                    FP += 1
                else:
                    TN += 1
        print('TP=%d FP=%d TN=%d FN=%d' % (TP, FP, TN, FN))
        return TP, FP, TN, FN

    def train(epoch, net, gen_set):
        train_loader = data.DataLoader(dataset=gen_set, shuffle=True, num_workers=num_worker, batch_size=batch,
                                       pin_memory=True)
        net.train()
        train_loss = 0
        totTP, totFP, totTN, totFN = 0, 0, 0, 0
        threshold = 0.5
        for batch_idx, (inputs, lb) in enumerate(tqdm(train_loader, desc="Iteration")):
            optimizer.zero_grad()
            inputs = inputs.float()
            lb = lb.float()
            valueseq = Var(inputs)
            lb = Var(lb)
            output = net(valueseq)
            if epoch > 110:
                aa = output.detach().cpu().numpy().reshape(-1)
                res = np.zeros(aa.shape, np.int64)
                res[aa > threshold] = 1
                bb = lb.detach().cpu().numpy().reshape(-1)
                TP, FP, TN, FN = calc(res, bb)
                totTP += TP
                totFP += FP
                totTN += TN
                totFN += FN
                if batch_idx % 100 == 0:
                    print('TP=%d FP=%d TN=%d FN=%d' % (TP, FP, TN, FN))
            loss1 = loss_function(output, lb)
            loss1.backward()
            train_loss += loss1.item()
            optimizer.step()
            torch.nn.utils.clip_grad_norm(net.parameters(), 5.0)
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(inputs), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                           loss1.item() / len(inputs)))

    model = Anomaly(win_size)
    net = model.cuda()
    gpu_num = torch.cuda.device_count()
    net = torch.nn.DataParallel(net, list(range(gpu_num)))
    print(net)
    base_lr = lr
    bp_parameters = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = optim.SGD(bp_parameters, lr=base_lr, momentum=0.9, weight_decay=0.0)

    if load_path != None:
        net = load_model(model, load_path)
        print("model loaded")

    gen_data = gen_set(win_size, data_path)
    for epoch in range(1, epochs + 1):
        print('epoch :', epoch)
        train(epoch, net, gen_data)
        adjust_lr(optimizer, epoch)
        if epoch % 5 == 0:
            save_model(model, model_path + 'srcnn_retry' + str(epoch) + '_' + str(win_size) + '.bin')
    return


def fft(values):
    wave = np.array(values)
    trans = np.fft.fft(wave)
    realnum = np.real(trans)
    comnum = np.imag(trans)
    mag = np.sqrt(realnum ** 2 + comnum ** 2)
    mag += 1e-5
    spectral = np.exp(np.log(mag) - average_filter(np.log(mag)))
    trans.real = trans.real * spectral / mag
    trans.imag = trans.imag * spectral / mag
    wave = np.fft.ifft(trans)
    mag = np.sqrt(wave.real ** 2 + wave.imag ** 2)
    return mag


def spectral_residual(values):
    """
    This method transform a time series into spectral residual series
    :param values: list.
        a list of float values.
    :return: mag: list.
        a list of float values as the spectral residual values
    """
    EPS = 1e-8
    trans = np.fft.fft(values)
    mag = np.sqrt(trans.real ** 2 + trans.imag ** 2)

    maglog = [np.log(item) if abs(item) > EPS else 0 for item in mag]

    spectral = np.exp(maglog - average_filter(maglog, n=3))

    trans.real = [ireal * ispectral / imag if abs(imag) > EPS else 0
                  for ireal, ispectral, imag in zip(trans.real, spectral, mag)]
    trans.imag = [iimag * ispectral / imag if abs(imag) > EPS else 0
                  for iimag, ispectral, imag in zip(trans.imag, spectral, mag)]

    wave_r = np.fft.ifft(trans)
    mag = np.sqrt(wave_r.real ** 2 + wave_r.imag ** 2)

    return mag


class gen_set(Dataset):
    def __init__(self, width, data_path):
        self.genlen = 0
        self.len = self.genlen
        self.width = width
        with open(data_path, 'r+') as fin:
            self.kpinegraw = json.load(fin)
        self.negrawlen = len(self.kpinegraw)
        print('length :', len(self.kpinegraw))
        self.len += self.negrawlen
        self.kpineglen = 0
        self.control = 0.

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        idx = index % self.negrawlen
        datas = self.kpinegraw[idx]
        datas = np.array(datas)
        data = datas[0, :].astype(np.float64)
        lbs = datas[1, :].astype(np.float64)
        wave = spectral_residual(data)
        waveavg = average_filter(wave)
        for i in range(self.width):
            if wave[i] < 0.001 and waveavg[i] < 0.001:
                lbs[i] = 0
                continue
            ratio = wave[i] / waveavg[i]
            if ratio < 1.0 and lbs[i] == 1:
                lbs[i] = 0
            if ratio > 5.0:
                lbs[i] = 1
        srscore = abs(wave - waveavg) / (waveavg + 0.01)
        sortid = np.argsort(srscore)
        for idx in sortid[-2:]:
            if srscore[idx] > 5:
                lbs[idx] = 1
        resdata = torch.from_numpy(100 * wave)
        reslb = torch.from_numpy(lbs)
        return resdata, reslb


def sr_cnn_eval(timestamp, value, label, window, net, ms_optioin, threshold=0.95, back_k=0, backaddnum=5, step=1):
    def Var(x):
        return Variable(x.cuda())

    def modelwork(x, net):
        with torch.no_grad():
            x = torch.from_numpy(100 * x).float()
            x = torch.unsqueeze(x, 0)
            x = Var(x)
            output = net(x)
        aa = output.detach().cpu().numpy().reshape(-1)
        res = np.zeros(aa.shape, np.int64)
        res[aa > threshold] = 1
        return res, aa

    win_size = window
    length = len(timestamp)
    if back_k <= 5:
        back = back_k
    else:
        back = 5
    detres = [0] * (win_size - backaddnum)
    scores = [0] * (win_size - backaddnum)

    for pt in range(win_size - backaddnum + back + step, length - back, step):
        head = max(0, pt - (win_size - backaddnum))
        tail = min(length, pt)
        wave = np.array(SpectralResidual.extend_series(value[head:tail + back]))
        mag = spectral_residual(wave)
        modeloutput, rawout = modelwork(mag, net)
        for ipt in range(pt - step - back, pt - back):
            detres.append(modeloutput[ipt - head])
            scores.append(rawout[ipt - head].item())
    detres += [0] * (length - len(detres))
    scores += [0] * (length - len(scores))

    if ms_optioin == 'anomaly':
        last = -1
        interval = min([timestamp[i] - timestamp[i - 1] for i in range(1, len(timestamp))])
        for i in range(1, len(timestamp)):
            if timestamp[i] - timestamp[i - 1] > interval:
                if last >= 0 and i - last < 1000:
                    detres[i] = 1
                    scores[i] = 1
            if detres[i] == 1:
                last = i

    return timestamp[:].tolist(), label[:], detres[:], scores[:]
