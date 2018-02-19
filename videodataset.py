from fastai.imports import *
from fastai.torch_imports import *
from fastai.core import *
from fastai.transforms import *
from fastai.layer_optimizer import *
from fastai.dataloader import DataLoader
from fastai.dataset import *


def read_dirs(path, folder):
    labels, video_dirs, all_labels = [], [], []
    full_path = os.path.join(path, folder)
    for label in sorted(os.listdir(full_path)):
        if not os.path.isdir(os.path.join(full_path, label)):
            continue
        all_labels.append(label)
        for video_dir in os.listdir(os.path.join(full_path, label)):
            video_dirs.append(os.path.join(folder, label, video_dir))
            labels.append(label)
    return video_dirs, labels, all_labels

def video_folder_source(path, folder):
    video_dirs, lbls, all_labels = read_dirs(path, folder)
    label2idx = {v:k for k,v in enumerate(all_labels)}
    idxs = [label2idx[lbl] for lbl in lbls]
    c = len(all_labels)
    label_arr = np.array(idxs, dtype=int)
    return video_dirs, label_arr, all_labels

class VideoBaseDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.n = self.get_n()
        self.c = self.get_c()
        self.sz = self.get_sz()

    def __getitem__(self, idx):
        x,y = self.get_x(idx),self.get_y(idx)
        return self.get(self.transform, x, y)

    def __len__(self): return self.n

    def get(self, tfm, x, y):
        if tfm is None:
            return (x, y)
        else:
            out = [tfm(img) for img in x]
            return out, y

    @abstractmethod
    def get_n(self): raise NotImplementedError
    @abstractmethod
    def get_c(self): raise NotImplementedError
    @abstractmethod
    def get_sz(self): raise NotImplementedError
    @abstractmethod
    def get_x(self, i): raise NotImplementedError
    @abstractmethod
    def get_y(self, i): raise NotImplementedError
    @property
    def is_multi(self): return True
    @property
    def is_reg(self): return False


class VideoFilesDataset(VideoBaseDataset):
    def __init__(self, video_dirs, transform, path):
        self.path,self.video_dirs = path,video_dirs
        super().__init__(transform)
    def get_n(self): return len(self.y)
    def get_sz(self): return self.transform.sz
    def get_x(self, i):
        video = []
        video_path = os.path.join(self.path, self.video_dirs[i])
        index = 0
        mod = 2
        for fname in os.listdir(video_path):
            if index%mod == 0:
                video.append(open_image(os.path.join(video_path, fname)))
            index += 1

        return video
    
    def resize_imgs(self, targ, new_path):
        dest = resize_imgs(self.fnames, targ, self.path, new_path)
        return self.__class__(self.fnames, self.y, self.transform, dest)

    def denorm(self,arr):
        """Reverse the normalization done to a batch of images.
        Arguments:
            arr: of shape/size (N,3,sz,sz)
        """
        if type(arr) is not np.ndarray: arr = to_np(arr)
        if len(arr.shape)==3: arr = arr[None]
        return self.transform.denorm(np.rollaxis(arr,1,4))

class VideoFilesArrayDataset(VideoFilesDataset):
    def __init__(self, video_dirs, y, transform, path):
        self.y=y
        assert(len(video_dirs)==len(y))
        super().__init__(video_dirs, transform, path)
    def get_y(self, i): return self.y[i]
    def get_c(self): return self.y.shape[1]

class VideoFilesIndexArrayDataset(VideoFilesArrayDataset):
    def get_c(self): return int(self.y.max())+1

class VideoClassifierData(ImageData):
    @property
    def is_multi(self): return self.trn_dl.dataset.is_multi

    @staticmethod
    def get_ds(fn, trn, val, tfms, test=None, **kwargs):
        res = [
            fn(trn[0], trn[1], tfms[0], **kwargs), # train
            fn(val[0], val[1], tfms[1], **kwargs), # val
            fn(trn[0], trn[1], tfms[1], **kwargs), # fix
            fn(val[0], val[1], tfms[0], **kwargs)  # aug
        ]
        if test is not None:
            test_lbls = np.zeros((len(test),1))
            res += [
                fn(test, test_lbls, tfms[1], **kwargs), # test
                fn(test, test_lbls, tfms[0], **kwargs)  # test_aug
            ]
        else: res += [None,None]
        return res

    @classmethod
    def from_paths(cls, path, bs=64, tfms=(None,None), trn_name='train', val_name='valid', test_name=None, num_workers=8):
        """ Read in videos and their labels given as sub-folder names
        Arguments:
            path: a root path of the data (used for storing trained models, precomputed values, etc)
            bs: batch size
            tfms: transformations (for data augmentations). e.g. output of `tfms_from_model`
            trn_name: a name of the folder that contains training videos.
            val_name:  a name of the folder that contains validation videos.
            test_name:  a name of the folder that contains test videos.
            num_workers: number of workers
        Returns:
            VideoClassifierData
        """
        trn,val = [video_folder_source(path, o) for o in (trn_name, val_name)]
        test_dirs = read_dirs(path, test_name) if test_name else None
        datasets = cls.get_ds(VideoFilesIndexArrayDataset, trn, val, tfms, path=path, test=test_dirs)
        return cls(path, datasets, bs, num_workers, classes=trn[2])