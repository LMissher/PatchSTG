import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import argparse
import numpy as np
import configparser
from tqdm import tqdm

from models.model import PatchSTG
from lib.utils import log_string, loadSpatialManagedIndices, _compute_loss, metric
from lib.data_loader import data_provider

class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)

        log_string(log, '\n------------ Loading Data -------------')
        self.train_data, self.train_loader = self._get_data(flag='train')
        _, self.vali_loader = self._get_data(flag='val')
        _, self.test_loader = self._get_data(flag='test')

        # ori_parts_idx indicates the original indices of input points
        # reo_parts_idx indicates the patched indices of input points
        # reo_all_idx indicates the patched indices of input and padded points
        self.ori_parts_idx, self.reo_parts_idx, self.reo_all_idx = loadSpatialManagedIndices(self.train_data, self.meta_file, self.adj_file,
                                                                                                self.recur_times, self.spa_patchsize, log)
        log_string(log, '------------ End -------------\n')

        self.best_epoch = 0

        self.device = torch.device(f"cuda:{self.cuda}" if torch.cuda.is_available() else "cpu")
        self.build_model()
    
    def build_model(self):
        self.model = PatchSTG(self.tem_patchsize, self.tem_patchnum,
                            self.node_num, self.spa_patchsize, self.spa_patchnum,
                            self.tod, self.dow,
                            self.layers, self.factors,
                            self.input_dims, self.node_dims, self.tod_dims, self.dow_dims,
                            self.ori_parts_idx, self.reo_parts_idx, self.reo_all_idx).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                        lr=self.learning_rate,weight_decay=self.weight_decay)

        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[1,35,40],
            gamma=0.5,
        )
    
    def _get_data(self, flag):
        # tod and dow means the time of the day and the day of the week
        data_set, data_loader = data_provider(self.num_workers, self.batch_size, self.traffic_file,
                                                self.train_ratio, self.test_ratio,
                                                self.input_len, self.output_len,
                                                self.tod, self.dow, flag, log)
        return data_set, data_loader

    def vali(self):
        self.model.eval()
        pred = []
        label = []

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_te, batch_y_te) in enumerate(self.vali_loader):
                if isinstance(self.model, torch.nn.Module):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_te = batch_x_te.to(self.device)

                    y_hat = self.model(batch_x, batch_x_te)

                    pred.append(self.train_data.inverse_transform(y_hat).cpu().numpy())
                    label.append(batch_y.cpu().numpy())
        
        pred = np.concatenate(pred, axis = 0)
        label = np.concatenate(label, axis = 0)

        maes = []
        rmses = []
        mapes = []

        for i in range(pred.shape[1]):
            mae, rmse , mape = metric(pred[:,i,:], label[:,i,:])
            maes.append(mae)
            rmses.append(rmse)
            mapes.append(mape)
            log_string(log,'step %d, mae: %.4f, rmse: %.4f, mape: %.4f' % (i+1, mae, rmse, mape))
        
        mae, rmse, mape = metric(pred, label)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        log_string(log, 'average, mae: %.4f, rmse: %.4f, mape: %.4f' % (mae, rmse, mape))
        
        return np.stack(maes, 0), np.stack(rmses, 0), np.stack(mapes, 0)

    def train(self):
        log_string(log, "======================TRAIN MODE======================")
        min_loss = 10000000.0

        for epoch in tqdm(range(1,self.max_epoch+1)):
            self.model.train()
            train_l_sum, batch_count, start = 0.0, 0, time.time()
            
            num_batch = math.ceil(len(self.train_data) / self.batch_size)
            with tqdm(total=num_batch) as pbar:
                for i, (batch_x, batch_y, batch_x_te, batch_y_te) in enumerate(self.train_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_te = batch_x_te.to(self.device)
                    
                    self.optimizer.zero_grad()

                    y_hat = self.model(batch_x, batch_x_te)

                    loss = _compute_loss(batch_y, self.train_data.inverse_transform(y_hat))
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                    self.optimizer.step()
                    
                    train_l_sum += loss.cpu().item()

                    batch_count += 1
                    pbar.update(1)

            log_string(log, 'epoch %d, lr %.6f, loss %.4f, time %.1f sec'
                % (epoch, self.optimizer.param_groups[0]['lr'], train_l_sum / batch_count, time.time() - start))
            mae, rmse, mape = self.vali()
            self.lr_scheduler.step()
            if mae[-1] < min_loss:
                self.best_epoch = epoch
                min_loss = mae[-1]
                torch.save(self.model.state_dict(), self.model_file)
        
        log_string(log, f'Best epoch is: {self.best_epoch}')

    def test(self):
        log_string(log, "======================TEST MODE======================")
        self.model.load_state_dict(torch.load(self.model_file, map_location=self.device))
        self.model.eval()
        pred = []
        label = []

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_te, batch_y_te) in enumerate(self.test_loader):
                if isinstance(self.model, torch.nn.Module):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_te = batch_x_te.to(self.device)

                    y_hat = self.model(batch_x, batch_x_te)

                    pred.append(self.train_data.inverse_transform(y_hat).cpu().numpy())
                    label.append(batch_y.cpu().numpy())
        
        pred = np.concatenate(pred, axis = 0)
        label = np.concatenate(label, axis = 0)

        maes = []
        rmses = []
        mapes = []

        for i in range(pred.shape[1]):
            mae, rmse , mape = metric(pred[:,i,:], label[:,i,:])
            maes.append(mae)
            rmses.append(rmse)
            mapes.append(mape)
            log_string(log,'step %d, mae: %.4f, rmse: %.4f, mape: %.4f' % (i+1, mae, rmse, mape))
        
        mae, rmse, mape = metric(pred, label)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        log_string(log, 'average, mae: %.4f, rmse: %.4f, mape: %.4f' % (mae, rmse, mape))
        
        return np.stack(maes, 0), np.stack(rmses, 0), np.stack(mapes, 0)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help='configuration file')
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)

    parser.add_argument('--num_workers', type = int, default = 32)
    parser.add_argument('--cuda', type=str, default=config['train']['cuda'])
    parser.add_argument('--seed', type = int, default = config['train']['seed'])
    parser.add_argument('--batch_size', type = int, default = config['train']['batch_size'])
    parser.add_argument('--max_epoch', type = int, default = config['train']['max_epoch'])
    parser.add_argument('--learning_rate', type=float, default = config['train']['learning_rate'])
    parser.add_argument('--weight_decay', type=float, default = config['train']['weight_decay'])

    parser.add_argument('--input_len', type = int, default = config['data']['input_len'])
    parser.add_argument('--output_len', type = int, default = config['data']['output_len'])
    parser.add_argument('--train_ratio', type = float, default = config['data']['train_ratio'])
    parser.add_argument('--val_ratio', type = float, default = config['data']['val_ratio'])
    parser.add_argument('--test_ratio', type = float, default = config['data']['test_ratio'])

    parser.add_argument('--layers', type=int, default = config['param']['layers'])
    parser.add_argument('--tem_patchsize', type = int, default = config['param']['tps'])
    parser.add_argument('--tem_patchnum', type = int, default = config['param']['tpn'])
    parser.add_argument('--factors', type=int, default = config['param']['factors'])
    parser.add_argument('--recur_times', type = int, default = config['param']['recur'])
    parser.add_argument('--spa_patchsize', type = int, default = config['param']['sps'])
    parser.add_argument('--spa_patchnum', type = int, default = config['param']['spn'])
    parser.add_argument('--node_num', type = int, default = config['param']['nodes'])
    parser.add_argument('--tod', type=int, default = config['param']['tod'])
    parser.add_argument('--dow', type=int, default = config['param']['dow'])
    parser.add_argument('--input_dims', type=int, default = config['param']['id'])
    parser.add_argument('--node_dims', type=int, default = config['param']['nd'])
    parser.add_argument('--tod_dims', type=int, default = config['param']['td'])
    parser.add_argument('--dow_dims', type=int, default = config['param']['dd'])

    parser.add_argument('--traffic_file', default = config['file']['traffic'])
    parser.add_argument('--meta_file', default = config['file']['meta'])
    parser.add_argument('--adj_file', default = config['file']['adj'])
    parser.add_argument('--model_file', default = config['file']['model'])
    parser.add_argument('--log_file', default = config['file']['log'])

    args = parser.parse_args()

    log = open(args.log_file, 'w')

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    
    log_string(log, '------------ Options -------------')
    for k, v in vars(args).items():
        log_string(log, '%s: %s' % (str(k), str(v)))
    log_string(log, '-------------- End ----------------')

    solver = Solver(vars(args))

    solver.train()
    solver.test()
    
