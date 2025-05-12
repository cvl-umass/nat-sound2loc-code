import os
import os.path as osp
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import json
import tqdm
import torchvision
from torchvision.transforms import v2
from torch.utils.data import default_collate
from torch.utils.data.sampler import Sampler
import h3 


from src import utils, config


class CustomMixup():
    def __init__(self, num_classes, batch_size, alpha=1.0):
        self.num_classes = num_classes
        self.alpha = alpha
        self.dist = torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([alpha]))
        self.flip_indices = torch.arange(batch_size-1, -1, -1)

    def transform(self, inpt, lam):
        # return torch.flip(inpt, [0]).mul_(1.0 - lam).add_(inpt.mul(lam))
        return inpt[self.flip_indices, :].mul_(1.0 - lam).add_(inpt.mul(lam))
        # return inpt.roll(1, 0).mul_(1.0 - lam).add_(inpt.mul(lam))

    def __call__(self, images, labels, geo):
        lam = self.dist.sample()
        mix_images = self.transform(images, lam)
        labels_oh = torch.nn.functional.one_hot(labels, num_classes=self.num_classes)
        mix_labels = self.transform(labels_oh.float(), lam)
        # geo = torch.cat([
        #     geo, geo[::-1, :], lam * torch.ones((geo.shape[0], 1))
        # ], dim=-1)
        return mix_images, mix_labels, geo, lam 

class MixupSampler(Sampler):
    def __init__(self, geo_locs, batch_size, resolution, balance=False):
        self.geo_locs = geo_locs
        self.batch_size = batch_size
        self.balance = balance
        assert self.batch_size % 2 == 0

        self.geo2idx = {}
        scale = torch.tensor([90, 180])
        for idx, d in enumerate(self.geo_locs):
            lat, lon = (d * scale).numpy().tolist()
            geo_cell = h3.latlng_to_cell(lat, lon, resolution)
            if geo_cell not in self.geo2idx:
                self.geo2idx[geo_cell] = []
            self.geo2idx[geo_cell].append(idx)
       
        self.all_geo_cells = list(self.geo2idx.keys())
        self.geo_cell_weights = [len(self.geo2idx[cell]) for cell in self.geo2idx]
        self.geo_cell_weights = [w / sum(self.geo_cell_weights) for w in self.geo_cell_weights]

    def __len__(self):
        return len(self.geo_locs)//self.batch_size

    def __iter__(self):
        for _ in range(len(self)):
            if self.balance:
                geo_cells = np.random.choice(self.all_geo_cells, self.batch_size // 2, replace=True)
            else:
                geo_cells = np.random.choice(self.all_geo_cells, self.batch_size // 2, p=self.geo_cell_weights, replace=True)
            
            batch = [None] * self.batch_size
            for cell_i, geo_cell in enumerate(geo_cells):
                all_indices = self.geo2idx[geo_cell]
                if len(all_indices) == 2:
                    batch[cell_i] = all_indices[0]
                    batch[-cell_i-1] = all_indices[1]
                elif len(all_indices) == 1:
                    batch[cell_i] = all_indices[0]
                    batch[-cell_i-1] = all_indices[0]
                else:   
                    id1, id2 = np.random.choice(all_indices, 2, replace=False)
                    batch[cell_i] = id1
                    batch[-cell_i-1] = id2
            # print(batch)
            assert None not in batch
            yield batch


        




class InatJsonDataset(Dataset):
    def __init__(self, args, task_type, data_root, split_path, transforms, geo_resolution, sound_aug=False, mode="test", window_len=512, test_stride=256):

        self.task_type = task_type
        self.transforms = transforms
        self.mode = mode
        self.sound_aug = sound_aug
        self.window_len = window_len
        self.test_stride = test_stride
        
        with open(split_path, "r") as f:
            dataset = json.load(f)
            
        self.class_name2idx = {
            c["audio_dir_name"]:c["id"]
            for c in dataset["categories"]
        }
        self.class_idx2name = {
            c["id"]:c["audio_dir_name"]
            for c in dataset["categories"]
        }
        
        audio_id2path = {
            au["id"]:au["file_name"]
            for au in dataset["audio"]
        }
        self.datapoints = [
            (
                osp.join(data_root, audio_id2path[a["audio_id"]].replace(".wav", ".npy")),
                self.class_idx2name[a["category_id"]],
                audio_id2path[a["audio_id"]].replace(".wav", "")
            )
            for a in dataset["annotations"]
        ]
        self.num_classes = len(list(self.class_name2idx.keys()))

        classes = [d[1] for d in self.datapoints]
        classes = sorted(list(set(classes)))
        class_idx = [self.class_name2idx[c] for c in classes]
        self.present_classes_mask = torch.zeros(self.num_classes)
        for c in class_idx:
            self.present_classes_mask[c] = 1
       

        audio_id2geo = {
            au["id"]: (au["latitude"], au["longitude"])
            for au in dataset["audio"]
        }
        scale = torch.tensor([90, 180])
        self.geo = [
            torch.Tensor(audio_id2geo[a["audio_id"]]) / scale
            for a in dataset["annotations"]
        ]

        task2dim = {
            "lat_lon": 2,
            "cartesian": 3,
            "classification": len(utils.resolution2classes(geo_resolution)),
            "geoclip": 512,
            "satclip": 256,
        }
        if self.task_type in  task2dim:
            self.num_geo_classes = task2dim[self.task_type]
        elif self.task_type == "generalclip":
            enc2dim = {
                "geoclip": 512,
                "satclip": 256,
                "sinr": 256
            }
            self.num_geo_classes = 0 
            for enc in args.loc_enc:
                self.num_geo_classes += enc2dim[enc]
        elif self.task_type == "audio_geoclip":
            if args.sinr_feat:
                self.num_geo_classes = 256 + 512
            else:
                self.num_geo_classes = 5547 + 512
        elif self.task_type == "geo_multitask":
            self.num_geo_classes = 0 
            assert args.tasks[0] == "geoclip"
            for t in args.tasks:
                self.num_geo_classes += task2dim[t]
        else:
            raise NotImplementedError


    def img_path2class(self, pth):
        return str(pth.split("/")[-2])

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):

        img_path, class_name, img_name = self.datapoints[idx]

        label = self.class_name2idx[class_name] if class_name is not None else -1

        img = np.load(img_path)
        img = np.stack([img]*3)

        if label == -1:
            img = (img - img.min()) * 0.5 + img.min()

        if self.mode == "train":
            img = self.time_selection(img)
            if self.sound_aug:
                img = self.time_masking(self.freq_masking(img))
        else:
            img, num = self.dense_prediction(img)
            label = torch.Tensor([label]*num).to(torch.long)
            

        img = torch.Tensor(img).to(torch.float) / 255.0
        if self.transforms is not None:
            img = self.transforms(img)

        geo = self.geo[idx] if self.geo is not None and idx < len(self.geo) else torch.Tensor([0, 0])

        return img, label, geo, [img_name]

    def dense_prediction(self, img):
        time_len = img.shape[-1]

        if time_len <= self.window_len:
            pad = self.window_len - time_len
            img = np.pad(img, ((0, 0), (0, 0), (pad//2, pad - pad//2)), constant_values=0)
            time_len = img.shape[-1]

        num = (time_len - self.window_len + self.test_stride - 1) // self.test_stride
        pad = self.test_stride * num - time_len + self.window_len
        img = np.pad(img, ((0, 0), (0, 0), (pad//2, pad - pad//2)), constant_values=0)
        
        out_list = []
        for i in range(num+1):
            start = i * self.test_stride
            if start + self.window_len > img.shape[-1]: break
            out_list.append(img[..., start : start + self.window_len])

        return np.stack(out_list), len(out_list)

    def time_selection(self, img):
        time_len = img.shape[-1]
        if time_len <= self.window_len:
            pad = self.window_len - time_len
            img = np.pad(img, ((0, 0), (0, 0), (pad//2, pad - pad//2)), constant_values=0)
            start = 0
        else:
            start = np.random.randint(0, time_len - self.window_len)
        return img[..., start : start + self.window_len]


    def freq_masking(self, img, mask_len=15):
        factor = np.random.RandomState().rand()
        freq_len = img.shape[-2]
        start = np.random.randint(0, freq_len - mask_len)
        interval = np.random.randint(0, mask_len)
        img[..., start : start + interval, :] = 0
        return img

    def time_masking(self, img, mask_len=15):
        time_len = img.shape[-1]
        start = np.random.randint(0, time_len - mask_len)
        interval = np.random.randint(0, mask_len)
        img[..., start : start + interval] = 0
        return img
    
            

standard_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.Normalize(
        (0.3342, 0.3342, 0.3342), (0.1529, 0.1529, 0.1529)
    ),
])

def read_sound(spec_path):
    img = np.load(spec_path)
    img = np.stack([img]*3, -1)
    window_len = 512
    test_stride = 256
    time_len = img.shape[-2]


    if time_len <= window_len:
        pad = window_len - time_len
        img = np.pad(img, ((0, 0), (pad//2, pad - pad//2), (0, 0)), constant_values=0)
        time_len = img.shape[-2]

    num = (time_len - window_len + test_stride - 1) // test_stride
    pad = test_stride * num - time_len + window_len
    if pad != 0:
        img = np.pad(img, ((0, 0), (0, pad), (0, 0)), constant_values=0)


    orig_frames = []
    frames_list = []
    # print(num, time_len)
    for i in range(num+1):
        start = i * test_stride
        end = start + window_len
        if end > img.shape[-2]: break
        # print(out_list[-1].shape, start, start+window_len)

        cur_img = img[..., start : end, :]
        orig_frames.append(cur_img)
        cur_img = torch.Tensor(cur_img).to(torch.float) / 255.0
        cur_img = cur_img.permute(2, 0, 1)
        cur_img = cur_img.unsqueeze(0)
        cur_img = standard_transforms(cur_img)
        frames_list.append(cur_img)

    return frames_list, orig_frames
      
        

def get_dataloaders(args):
    standard_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.Normalize(
            (0.3342, 0.3342, 0.3342), (0.1529, 0.1529, 0.1529)
        ),
    ])

    train_transforms = standard_transforms
    sound_aug = True if hasattr(args, "sound_aug") and args.sound_aug else False

    
    dataset_class = InatJsonDataset
    data = dataset_class(
        args=args,
        task_type=args.task_type,
        data_root=config.SPECTROGRAM_DIR,
        split_path=config.TRAIN_JSON,
        transforms = train_transforms,
        geo_resolution = args.geo_resolution,
        sound_aug=sound_aug,
        mode="train",
    )
    val_data = dataset_class(
        args=args,
        task_type=args.task_type,
        data_root=config.SPECTROGRAM_DIR,
        split_path=config.VAL_JSON,
        transforms = standard_transforms,
        geo_resolution = args.geo_resolution,
        sound_aug=False,
        mode="test",
    )
    test_data = dataset_class(
        args=args,
        task_type=args.task_type,
        data_root=config.SPECTROGRAM_DIR,
        split_path=config.TEST_JSON,
        transforms = standard_transforms,
        geo_resolution = args.geo_resolution,
        sound_aug=False,
        mode="test",
    )


    num_classes = data.num_classes
    num_geo_classes = data.num_geo_classes
    if hasattr(args, "mixup") and args.mixup:
        mixup = CustomMixup(num_classes=num_classes, batch_size=args.batch_size, alpha=args.mixup_alpha)
        def mixup_collate(batch):
            collated_batch = default_collate(batch)
            images, labels, geo = collated_batch[:3]
            images_orig = images.clone().detach()
            images = torch.exp(images - 10)
            images_mix, labels_mix, geo, lam = mixup(images, labels, geo)
            images_mix = torch.log(images_mix) + 10

            mixed_batch = (images_mix, labels_mix, geo, *collated_batch[3:])
            return mixed_batch
        collate_fn = mixup_collate
        batch_sampler = MixupSampler(data.geo, args.batch_size, args.geo_resolution, balance=args.class_balance)
        train_dataloader = DataLoader(data, collate_fn=collate_fn, batch_sampler=batch_sampler, num_workers=4)
    else:
        collate_fn = default_collate
        sampler = torch.utils.data.sampler.RandomSampler(data)
        train_dataloader = DataLoader(data, batch_size=args.batch_size, collate_fn=collate_fn, sampler=sampler, num_workers=4)
        
    
    def test_collate(batch):

        images = [b[0] for b in batch]
        labels = [b[1] for b in batch]
        geo = [b[2] for b in batch]
        img_names = [b[3] for b in batch]

        all_img_names = []
        for i, n in enumerate(img_names):
            all_img_names.extend(n*images[i].shape[0])
        all_geo = []
        for i, g in enumerate(geo):
            all_geo.extend([g]*images[i].shape[0])
        images = torch.cat(images, 0)
        labels = torch.cat(labels, 0)
        geo = torch.stack(all_geo, 0)
        # geo =  torch.Tensor(all_geo).to(torch.long)
        full_batch = (images, labels, geo, all_img_names)
        return full_batch

    val_dataloader = DataLoader(val_data, batch_size=4, shuffle=False, collate_fn=test_collate)
    test_dataloader = DataLoader(test_data, batch_size=4, shuffle=False, collate_fn=test_collate)

    train_dataloader.num_classes = num_classes
    val_dataloader.num_classes = num_classes
    test_dataloader.num_classes = num_classes

    train_dataloader.num_geo_classes = num_geo_classes
    val_dataloader.num_geo_classes = num_geo_classes
    test_dataloader.num_geo_classes = num_geo_classes

    return train_dataloader, val_dataloader, test_dataloader
