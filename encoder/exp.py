from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
import time
import warnings
import numpy as np
import pickle as pkl
import pdb
from sklearn.metrics import f1_score

from model import Model

warnings.filterwarnings('ignore')

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        
        self.device = self._acquire_device()
        self._build_model()

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
    
    
class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)
        self.model = Model(args, self.device).float().to(self.device)

    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        try:
            self.args.enc_in = train_data.feature_df.shape[1]
        except:
            self.args.enc_in = train_data.num_feature
        self.args.num_class = len(train_data.class_names)
        
        # model init
        model = Model(self.args, self.device).float()

        #return model_time, model_text

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        try:
            self.class_freq = data_set.class_freq
        except:
            pass
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        """
        weights = 1. / self.class_freq
        weights = weights / weights.sum()
        weights = torch.from_numpy(weights).float().to(self.device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print('weighted cross entropy')
        """
        criterion = nn.CrossEntropyLoss()
        return criterion
    
    def cal_c_loss(self, pos, aug, c_temp=0.2):
        pos = F.normalize(pos, p=2, dim=1)
        aug = F.normalize(aug, p=2, dim=1)
        pos_score = torch.sum(pos * aug, dim=1) # [batch_size]
        ttl_score = torch.matmul(pos, aug.permute(1, 0)) # [batch_size, batch_size]

        pos_score = torch.exp(pos_score / c_temp) # [batch_size]
        ttl_score = torch.sum(torch.exp(ttl_score / c_temp), axis=1) # [batch_size]

        c_loss = - torch.mean(torch.log(pos_score / ttl_score))

        return c_loss

    def vali(self, vali_data, vali_loader, criterion):
        trues, preds = [], []
        
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x_time, batch_x_text, label) in enumerate(vali_loader):
                batch_x_time = batch_x_time.float().to(self.device)
                label = label.to(self.device)

                outputs, _ = self.model(batch_x_time, batch_x_text)
        
                preds.append(outputs.detach())
                trues.append(label)

        trues = torch.cat(trues, 0)
        trues = trues.flatten().cpu().numpy()
        
        preds = torch.cat(preds, 0)
        probs = torch.nn.functional.softmax(preds) 
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  
        
        f1_micro = f1_score(trues, predictions, average='micro')
        f1_macro = f1_score(trues, predictions, average='macro')

        self.model.train()
        
        return f1_micro, f1_macro

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')
        try:
            vali_data, vali_loader = self._get_data(flag='VAL')
        except:
            vali_data, vali_loader = self._get_data(flag='TEST')
        test_data, test_loader = self._get_data(flag='TEST')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x_time, batch_x_text, label) in enumerate(train_loader):
                if len(label) <= 1:
                    continue
                iter_count += 1
                model_optim.zero_grad()

                batch_x_time = batch_x_time.float().to(self.device)
                label = label.to(self.device)

                outputs, _ = self.model(batch_x_time, batch_x_text)
                
                batch_y = label.long().squeeze(-1)
                if self.args.loss_type == 'balance':
                    loss = []
                    for c in range(self.args.num_class):
                        idx_c = batch_y == c
                        _loss = criterion(outputs[idx_c], batch_y[idx_c]).mean()
                        if torch.sum(idx_c) > 0:
                            loss.append(_loss)
                    loss = sum(loss) / len(loss)
                elif self.args.loss_type == 'naive':
                    loss = criterion(outputs, batch_y).mean()
                
                train_loss.append(loss.item())
                
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()
                
            print("\nEpoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            
            train_loss = np.average(train_loss)
            val_micro, val_macro = self.vali(vali_data, vali_loader, criterion)
            test_micro, test_macro = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.3f}\nVali-Micro: {3:.3f} Vali-Macro: {4:.3f}\nTest-Micro: {5:.3f} Test-Macro: {6:.3f}".format(epoch + 1, train_steps, train_loss, val_micro, val_macro, test_micro, test_macro))
            
            early_stopping(-val_macro, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if (epoch + 1) % 5 == 0:
                adjust_learning_rate(model_optim, epoch + 1, self.args)
            
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def test(self, setting, test=0):
        all_data, all_loader = self._get_data(flag='ALL')
        
        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        embs, trues, preds, probs = [], [], [], []
        
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x_time, batch_x_text, label) in enumerate(all_loader):
                batch_x_time = batch_x_time.float().to(self.device)
                label = label.to(self.device)

                outputs, emb = self.model(batch_x_time, batch_x_text)

                trues.append(label.detach().cpu())
                preds.append(torch.max(outputs, dim=1)[1].detach().cpu())
                probs.append(outputs.detach().cpu())
                embs.append(emb.detach().cpu())

        trues = torch.cat(trues).numpy()
        preds = torch.cat(preds).numpy()
        probs = F.softmax(torch.cat(probs, 0)).numpy()

        f1_micro = f1_score(trues, preds, average='micro')
        f1_macro = f1_score(trues, preds, average='macro')

        print(f'F1-Micro: {f1_micro}')
        print(f'F1-Macro: {f1_macro}')
        print()
        
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        np.save(folder_path + 'metrics.npy', np.array([f1_micro, f1_macro]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'prob.npy', probs)
        
        
        #embs = torch.stack(embs).squeeze(1)
        embs = torch.cat(embs, 0)
        
        with open(f'embeddings/{setting}.pkl', 'wb') as f:
            pkl.dump(embs, f)
            
        # result save
        os.remove(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))

        return