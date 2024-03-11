import os
import json
from tensorboardX import SummaryWriter

exp_path_list = ['exp']
log_keys = ['train_lr', 'train_loss', 'test_loss', 'test_acc1', 'test_acc5']

for path in exp_path_list:
    for exp in os.listdir(path):
        log_path = os.path.join('.', path, exp, 'ckpt', 'log.txt')
        if os.path.exists(log_path):
            tensorboard_path = os.path.join('.', path, exp, 'events')
            if os.path.exists(tensorboard_path):
                for old_exp in os.listdir(tensorboard_path):
                    delete_path = os.path.join(tensorboard_path, old_exp)
                    print('delete:', delete_path)
                    os.remove(delete_path)
            tb_logger = SummaryWriter(tensorboard_path)
            with open(log_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    log = json.loads(line.rstrip())
                    for k in log_keys:
                        tb_logger.add_scalar(k, log[k], log['epoch'])
            print("load ok in:", tensorboard_path)
            tb_logger.close()
        log_path = os.path.join('.', path, exp, 'log', 'log.txt')
        if os.path.exists(log_path):
            tensorboard_path = os.path.join('.', path, exp, 'events')
            if os.path.exists(tensorboard_path):
                for old_exp in os.listdir(tensorboard_path):
                    delete_path = os.path.join(tensorboard_path, old_exp)
                    print('delete:', delete_path)
                    os.remove(delete_path)
            tb_logger = SummaryWriter(tensorboard_path)
            with open(log_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    log = json.loads(line.rstrip())
                    for k in log_keys:
                        tb_logger.add_scalar(k, log[k], log['epoch'])
            print("load ok in:", tensorboard_path)    
            tb_logger.close()
