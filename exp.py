
import os
import os.path as osp
import json
import torch
import pickle
import logging
import numpy as np
from model import SimVP
from tqdm import tqdm
from API import *
from utils import *


class Exp:
    def __init__(self, args):
        super(Exp, self).__init__()
        self.args = args
        self.config = self.args.__dict__
        #self.device = self._acquire_device()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._preparation()
        print_log(output_namespace(self.args))

        self._get_data()
        self._select_optimizer()
        self._select_criterion()

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:{}'.format(0))
            print_log('Use GPU: {}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print_log('Use CPU')
        return device

    def _preparation(self):
        # seed
        set_seed(self.args.seed)
        # log and checkpoint
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')
        # prepare data
        self._get_data()
        # build the model
        self._build_model()

    def _build_model(self):
        print("building model")
        args = self.args
        self.model = SimVP(tuple(args.in_shape), args.hid_S,
                           args.hid_T, args.N_S, args.N_T).to(self.device)
        if args.resume is not None:        
            self.model.dec.readout = torch.nn.Identity()
            self.model.load_state_dict(torch.load(args.resume), strict = False)
        if self.args.dataname == 'dl_seg':
            self.model.dec.readout = torch.nn.Conv2d(self.args.hid_S, 49, kernel_size=(1, 1), stride=(1, 1)).to(self.device)

    def _get_data(self):
        config = self.args.__dict__
        self.train_loader, self.vali_loader, self.test_loader, self.data_mean, self.data_std = load_data(**config)
        self.vali_loader = self.test_loader if self.vali_loader is None else self.vali_loader

    def _select_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr)
        #self.optimizer = torch.optim.Adam(
        #    self.model.hid.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.args.lr, steps_per_epoch=len(self.train_loader), epochs=self.args.epochs)
        return self.optimizer

    def _select_criterion(self):
        weight = torch.ones(49).to(self.device)
        weight[0] = 0.3
        self.criterion = torch.nn.CrossEntropyLoss(weight = weight) if self.args.dataname == 'dl_seg' else torch.nn.MSELoss()

    def _save(self, name=''):
        print(os.path.join(self.checkpoints_path, name + '.pth'))
        torch.save(self.model.state_dict(), os.path.join(
            self.checkpoints_path, name + '.pth'))
        state = self.scheduler.state_dict()
        fw = open(os.path.join(self.checkpoints_path, name + '.pkl'), 'wb')
        pickle.dump(state, fw)

    def train(self, args):
        config = args.__dict__
        recorder = Recorder(verbose=True)

        for epoch in range(config['epochs']):
            self.test(args)
            train_loss = []
            self.model.train()
            train_pbar = tqdm(self.train_loader)

            for batch_x, batch_y in train_pbar:
                self.optimizer.zero_grad()
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_y = self.model(batch_x, args.dataname == 'dl_seg')
                
                if self.args.dataname == 'dl_seg':
                    pred_y = pred_y.permute(0, 1, 3, 4, 2).reshape(-1, 49)
                    batch_y = batch_y.reshape(-1)
                    loss = self.criterion(pred_y, batch_y)
                else:
                    loss = 0.25 * self.criterion(pred_y[:, -1], batch_y[:, -1]) + self.criterion(pred_y, batch_y)
                train_loss.append(loss.item())
                train_pbar.set_description('train loss: {:.8f}'.format(loss.item()))

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            train_loss = np.average(train_loss)

            if epoch % args.log_step == 0:
                self._save(name=str(epoch))
                with torch.no_grad():
                    vali_loss = self.vali(self.vali_loader)
                print_log("Epoch: {0} | Train Loss: {1:.8f} Vali Loss: {2:.8f}\n".format(
                    epoch + 1, train_loss, vali_loss))
                recorder(vali_loss, self.model, self.path)
                
        best_model_path = self.path + '/' + 'checkpoint.pth'
        print(best_model_path)
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def vali(self, vali_loader):
        self.model.eval()
        preds_lst, trues_lst, total_loss = [], [], []
        vali_pbar = tqdm(vali_loader)
        for i, (batch_x, batch_y) in enumerate(vali_pbar):

            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            pred_y = self.model(batch_x, self.args.dataname == 'dl_seg')
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 pred_y, batch_y], [preds_lst, trues_lst]))

            if self.args.dataname == 'dl_seg':
                pred_y = pred_y.permute(0, 1, 3, 4, 2).reshape(-1, 49)
                batch_y = batch_y.reshape(-1)
                loss = self.criterion(pred_y, batch_y)
            else:
                loss = 0.7 * self.criterion(pred_y[:, -1], batch_y[:, -1]) + 0.3 * self.criterion(pred_y, batch_y)
            vali_pbar.set_description(
                'vali loss: {:.4f}'.format(loss.mean().item()))
            total_loss.append(loss.mean().item())

        total_loss = np.average(total_loss)
        print(total_loss)
        #mse, mae, ssim, psnr = metric(preds, trues, self.data_mean, self.data_mean, True)
        if self.args.dataname != 'dl_seg':
            #preds = torch.argmax(torch.tensor(preds), dim = -1)
            #preds = preds.unsqueeze(2).numpy()
            #trues = trues[:, :, np.newaxis, :, :]
            preds = np.concatenate(preds_lst, axis=0)
            trues = np.concatenate(trues_lst, axis=0)
            mse, mae = metric(preds, trues, self.data_mean, self.data_std, False)
            #print_log('vali mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}'.format(mse, mae, ssim, psnr))
            print_log('vali mse:{:.8f}, mae:{:.8f}'.format(mse, mae))
        self.model.train()
        return total_loss

    def test(self, args):
        self.model.eval()
        inputs_lst, trues_lst, preds_lst = [], [], []
        for i, (batch_x, batch_y) in enumerate(self.test_loader):
            if i > 25:
                break
            pred_y = self.model(batch_x.to(self.device), self.args.dataname == 'dl_seg')
            if self.args.dataname == 'dl_seg':
                pred_y = pred_y.permute(0, 1, 3, 4, 2)
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 batch_x, batch_y, pred_y], [inputs_lst, trues_lst, preds_lst]))

        inputs, trues, preds = map(lambda data: np.concatenate(
            data, axis=0), [inputs_lst, trues_lst, preds_lst])

        folder_path = self.path+'/results/{}/sv/'.format(args.ex_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        print(preds.shape, trues.shape)
        #mse, mae, ssim, psnr = metric(preds, trues, self.test_loader.dataset.mean, self.test_loader.dataset.std, True)
        if args.dataname == 'dl_seg':
            preds = torch.argmax(torch.tensor(preds), dim = -1)
            preds = preds.unsqueeze(2).numpy()
            trues = trues[:, :, np.newaxis, :, :]
        mse, mae = metric(preds, trues, self.data_mean, self.data_std, False)
        #print_log('mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}'.format(mse, mae, ssim, psnr))
        print_log('vali mse:{:.4f}, mae:{:.4f}'.format(mse, mae))

        for np_data in ['inputs', 'trues', 'preds']:
            print(osp.join(folder_path, np_data + '.npy'))
            np.save(osp.join(folder_path, np_data + '.npy'), vars()[np_data])
        return mse
