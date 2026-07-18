import glob
import os

from data import srdata


class HRLRPngDataset(srdata.SRData):
    def __init__(self, args, root_dir, train=True):
        super(HRLRPngDataset, self).__init__(args, root_dir, train)
        self._set_filesystem()
        self.train = train
        self.scale = args.scale[0]

    def _set_filesystem(self):
        self.dir_hr = os.path.join(self.root_dir, 'HR')
        self.dir_lr = os.path.join(self.root_dir, 'LR_x' + str(self.scale))
        self.ext = '.png'

    def _scan(self):
        list_hr = sorted(glob.glob(os.path.join(self.dir_hr, '*' + self.ext)))
        list_lr = []
        for hr_path in list_hr:
            filename = os.path.split(hr_path)[-1]
            list_lr.append(os.path.join(self.dir_lr, filename))
        return list_hr, list_lr

    def __len__(self):
        return len(self.hr_img_dirs)
