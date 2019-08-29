import argparse
from utils import *
import numpy as np
import os
import time


def auto(epoch):
    path_auto = os.getcwd() + '/auto.json'
    with open(path_auto, 'r+') as f:
        store = json.load(f)
    data = store['data']
    window = store['window']
    store['epoch'] = epoch
    with open(path_auto, 'w+') as f:
        json.dump(store, f)
    return data, window


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SRCNN')
    parser.add_argument('--data', type=str, default='kensho-min-6000', help='location of the data file')
    parser.add_argument('--window', type=int, default=128, help='window size')
    parser.add_argument('--lr', type=int, default=1e-6, help='learning rate')
    parser.add_argument('--step', type=int, default=64, help='step')

    parser.add_argument('--seed', type=int, default=54321, help='random seed')
    parser.add_argument('--load', type=bool, default=False, help='load the existed model')
    parser.add_argument('--save', type=str, default='snapshot', help='path to save the model')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256, help='path to save the model')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers of pytorch')
    parser.add_argument('--model', type=str, default='sr_cnn', help='model')
    parser.add_argument('--auto', type=bool, default=False, help='Automatic filling parameters')

    args = parser.parse_args()
    # seed,data,window,save ,load,lr,epoch,batch_size,num_workers
    # data step window seed
    if args.auto:
        data, window = auto(args.epoch)
    else:
        data, window = args.data, args.window
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    models = {
        'sr_cnn': sr_cnn,
    }
    model = args.model
    root_path = os.getcwd()
    train_data_path = root_path + '/' + data + '_' + str(window) + '_train.json'
    model_path = root_path + '/' + args.save + '/'
    if args.load:
        load_path = root_path + '/' + args.load
    else:
        load_path = None

    total_time = 0
    time_start = time.time()
    # making training here
    models[model](train_data_path, model_path, window, args.lr, args.epoch, args.batch_size, args.num_workers,
                  load_path=load_path)
    time_end = time.time()
    total_time += time_end - time_start
    print('time used for training:', total_time, 'seconds')