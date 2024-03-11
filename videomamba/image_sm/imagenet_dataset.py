import os.path as osp
import json
import requests
import time
import numpy as np
import io
from PIL import Image
import logging
import random
import torch
from torch.utils.data import Dataset
logger = logging.getLogger('global')

try:
    from petrel_client.client import Client
    has_client = True
except ImportError:
    has_client = False


def pil_loader(img_bytes, filepath):
    buff = io.BytesIO(img_bytes)
    try:
        with Image.open(buff) as img:
            img = img.convert('RGB')
    except IOError:
        logger.info('Failed in loading {}'.format(filepath))
    return img


class ImageNetDataset(Dataset):
    """
    ImageNet Dataset.

    Arguments:
        - root_dir (:obj:`str`): root directory of dataset
        - meta_file (:obj:`str`): name of meta file
        - transform (list of ``Transform`` objects): list of transforms
        - read_type (:obj:`str`): read type from the original meta_file
        - evaluator (:obj:`Evaluator`): evaluate to get metrics
        - image_reader_type (:obj:`str`): reader type 'pil' or 'ks'
        - server_cfg (list): server configurations

    Metafile example::
        "n01440764/n01440764_10026.JPEG 0\n"
    """
    def __init__(self, root_dir, meta_file, transform=None,
                 read_from='fs', evaluator=None,
                 server_cfg={}):

        self.root_dir = root_dir
        self.meta_file = meta_file
        self.read_from = read_from
        self.transform = transform
        self.evaluator = evaluator
        self.image_reader = pil_loader
        self.initialized = False
        self.use_server = False

        if len(server_cfg) == 0:
            # read from local file
            if meta_file.startswith('s3://'):
                f = Client().Get(meta_file).decode('utf-8').strip().split('\n')
                lines = f
            else:
                f = open(meta_file, 'r')     
                if f:
                    lines = f.readlines()

            self.num = len(lines)
            self.metas = []
            for line in lines:
                filename, label = line.rstrip().split()
                self.metas.append({'filename': filename, 'label': label})
        else:
            # read from http server
            self.server_ip = server_cfg['ip']
            self.server_port = server_cfg['port']
            self.use_server = True
            if isinstance(self.server_ip, str):
                self.server_ip = [self.server_ip]
            if isinstance(self.server_port, int):
                self.server_port = [self.server_port]
            assert len(self.server_ip) == len(self.server_port), \
                'length of ips should equal to the length of ports'

            self.num = int(requests.get('http://{}:{}/get_len'.format(
                self.server_ip[0], self.server_port[0])).json())

    def __len__(self):
        return self.num

    def _load_meta(self, idx):
        if self.use_server:
            while True:
                # random select a server ip
                rdx = np.random.randint(len(self.server_ip))
                r_ip, r_port = self.server_ip[rdx], self.server_port[rdx]
                # require meta information
                try:
                    meta = requests.get('http://{}:{}/get/{}'.format(r_ip, r_port, idx), timeout=500).json()
                    break
                except Exception:
                    time.sleep(0.005)

            return meta
        else:
            return self.metas[idx]

    def _init_petrel(self):
        if not self.initialized:
            assert has_client==True, 'petrel_client are not installed'
            self.client = Client("~/petreloss.conf")
            self.initialized = True

    def read_file(self, meta_dict):
        filebytes = None
        if self.read_from == 'fake':
            if self.initialized:
                filebytes = self.saved_filebytes
            else:
                filebytes = self.saved_filebytes = np.fromfile(meta_dict['filename'], dtype=np.uint8)
                self.initialized = True
        elif self.read_from == 'petrel' or 's3://' in meta_dict['filename']:
            self._init_petrel()
            value = self.client.Get(meta_dict['filename'])
            filebytes = np.frombuffer(value, dtype=np.uint8)
        elif self.read_from == 'fs':
            filebytes = np.fromfile(meta_dict['filename'], dtype=np.uint8)
        else:
            raise RuntimeError("unknown value for read_from: {}".format(self.read_from))
        return filebytes

    def __getitem__(self, idx):
        try:
            curr_meta = self._load_meta(idx)
            filename = osp.join(self.root_dir, curr_meta['filename'])
            label = int(curr_meta['label'])
            # add root_dir to filename
            curr_meta['filename'] = filename
            img_bytes = self.read_file(curr_meta)
            img = self.image_reader(img_bytes, filename)

            if self.transform is not None:
                img = self.transform(img)

            # item = {
            #     'image': img,
            #     'label': label,
            #     'image_id': idx,
            #     'filename': filename
            # }
            # return item
            return img, label
        except Exception:
            return self.__getitem__(random.randint(0, self.num - 1))

    def tensor2numpy(self, x):
        if x is None:
            return x
        if torch.is_tensor(x):
            return x.cpu().numpy()
        if isinstance(x, list):
            x = [_.cpu().numpy() if torch.is_tensor(_) else _ for _ in x]
        return x

    def dump(self, writer, output):
        prediction = self.tensor2numpy(output['prediction'])
        label = self.tensor2numpy(output['label'])
        score = self.tensor2numpy(output['score'])

        if 'filename' in output:
            # pytorch type: {'image', 'label', 'filename', 'image_id'}
            filename = output['filename']
            image_id = output['image_id']
            for _idx in range(prediction.shape[0]):
                res = {
                    'filename': filename[_idx],
                    'image_id': int(image_id[_idx]),
                    'prediction': int(prediction[_idx]),
                    'label': int(label[_idx]),
                    'score': [float('%.8f' % s) for s in score[_idx]],
                }
                writer.write(json.dumps(res, ensure_ascii=False) + '\n')
        else:
            # dali type: {'image', 'label'}
            for _idx in range(prediction.shape[0]):
                res = {
                    'prediction': int(prediction[_idx]),
                    'label': int(label[_idx]),
                    'score': [float('%.8f' % s) for s in score[_idx]],
                }
                writer.write(json.dumps(res, ensure_ascii=False) + '\n')
        writer.flush()
