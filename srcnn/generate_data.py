import argparse
# from pluskensho_gen_fake_train import *
from utils import *
import os
import time
from util import average_filter


class gen():
    def __init__(self, win_siz, step, nums):
        self.control = 0
        self.win_siz = win_siz
        self.step = step
        self.number = nums

    def generate_train_data(self, value, back_k=0):
        def normalize(a):
            amin = np.min(a)
            amax = np.max(a)
            a = (a - amin) / (amax - amin + 1e-5)
            return 3 * a

        if back_k <= 5:
            back = back_k
        else:
            back = 5
        length = len(value)
        tmp = []
        for pt in range(self.win_siz, length - back, self.step):
            head = max(0, pt - self.win_siz)
            tail = min(length - back, pt)
            # print("length",length,win_size,pt,head,tail,len(detres))
            # gen data
            data = np.array(value[head:tail])
            data = data.astype(np.float64)
            data = normalize(data)
            num = np.random.randint(1, self.number)
            ids = np.random.choice(self.win_siz, num, replace=False)
            lbs = np.zeros(self.win_siz, dtype=np.int64)
            if (self.win_siz - 6) not in ids:
                self.control += np.random.random()
            else:
                self.control = 0
            if self.control > 100:
                ids[0] = self.win_siz - 6
                self.control = 0
            mean = np.mean(data)
            dataavg = average_filter(data)
            var = np.var(data)
            for id in ids:
                data[id] += (dataavg[id] + mean) * np.random.randn() * min((1 + var), 10)
                lbs[id] = 1
            tmp.append([data.tolist(), lbs.tolist()])
        return tmp


def auto(dic):
    # automatic generate parameters
    path_auto = os.getcwd() + '/auto.json'
    auto = {}
    for item, value in dic:
        if value != None:
            auto[item] = value
    with open(path_auto, 'w+') as f:
        json.dump(auto, f)


def get_path(data):
    dir_ = os.getcwd() + '/' + data + '/'
    fadir = [_ for _ in os.listdir(dir_)]
    print(fadir, 'fadir')
    files = []
    for eachdir in fadir:
        files += [dir_ + eachdir + '/' + _ for _ in os.listdir(dir_ + eachdir)]
    print(files, 'files')
    return files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SRCNN')
    parser.add_argument('--data', type=str, required=True, help='location of the data file')
    parser.add_argument('--window', type=int, default=128, help='window size')
    parser.add_argument('--step', type=int, default=64, help='step')
    parser.add_argument('--seed', type=int, default=54321, help='random seed')
    parser.add_argument('--num', type=int, default=10, help='upper limit value for the number of anomaly points')
    args = parser.parse_args()
    np.random.seed(args.seed)
    auto(vars(args).items())
    files = get_path(args.data)

    train_data_path = os.getcwd() + '/' + args.data + '_' + str(args.window) + '_train.json'
    total_time = 0
    results = []
    print("generating train data")
    generator = gen(args.window, args.step, args.num)
    for f in files:
        print('reading', f)
        in_timestamp, in_value = read_csv(f)
        in_label = []
        # the timestamp, value, and label for one series
        # in_timestamp, in_value, in_label = tmp_data['timestamp'], tmp_data['value'], tmp_data['label']
        if len(in_value) < args.window:
            print("value's length < window size", len(in_value), args.window)
            continue
        time_start = time.time()
        train_data = generator.generate_train_data(in_value)
        # print('traindata :',len(train_data))
        time_end = time.time()
        total_time += time_end - time_start
        results += train_data
        # print(np.array(train_data).shape,np.array(results).shape,'results')
    print('file num:', len(files))
    print('total fake data size:', len(results))
    with open(train_data_path, 'w+') as f:
        print(train_data_path)
        json.dump(results, f)