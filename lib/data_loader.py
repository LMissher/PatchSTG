import numpy as np
from torch.utils.data import Dataset, DataLoader
from .utils import log_string

class Dataset_Custom(Dataset):
    def __init__(self, data_path, tro, iro, P, Q, tod, dow, flag):
        self.seq_len = P
        self.pred_len = Q
        self.train_ratio = tro
        self.test_ratio = iro
        self.tod = tod
        self.dow = dow
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        df_raw = np.load(self.data_path)['data'][...,:1]
        time_steps = df_raw.shape[0]

        num_train = int(len(df_raw) * self.train_ratio)
        num_test = int(len(df_raw) * self.test_ratio)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        train_data = df_raw[border1s[0]:border2s[0]]
        self.mean, self.std = np.mean(train_data), np.std(train_data)
        data = (df_raw - self.mean) / self.std

        self.data_x = data[border1:border2]
        self.data_y = df_raw[border1:border2]

        data_stamp = np.zeros([time_steps, 2])
        data_stamp[:,0] = np.array([i % self.tod for i in range(time_steps)])
        data_stamp[:,1] = np.array([(i // self.tod) % self.dow for i in range(time_steps)])
        data_stamp = np.repeat(np.expand_dims(data_stamp, 1), df_raw.shape[1], 1)

        self.data_stamp = data_stamp[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_te = self.data_stamp[s_begin:s_end]
        seq_y_te = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_te, seq_y_te

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def data_provider(num_workers, batch_size, data_path, tro, iro, P, Q, tod, dow, flag, log):
    Data = Dataset_Custom

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = batch_size

    data_set = Data(
        data_path=data_path,
        tro=tro,
        iro=iro,
        P=P,
        Q=Q,
        tod=tod,
        dow=dow,
        flag=flag,
    )
    log_string(log, f'{flag}: {len(data_set)}')
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=num_workers,
        drop_last=drop_last
    )
    return data_set, data_loader