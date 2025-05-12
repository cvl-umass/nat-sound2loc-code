import warnings
warnings.filterwarnings("ignore")

import matplotlib
viridis = matplotlib.colormaps['viridis']
plasma = matplotlib.colormaps['plasma']
import cv2
import argparse
import torchvision
import torch
import cv2
import tqdm
import json
import copy
import h3 
import numpy as np
import os
from dataset import read_sound
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

from multiprocessing import Process
from geoclip import LocationEncoder


from src import models, utils, config



def ag_clip_model_wrapper(model, loc_dim=512):
    def wrapper(x):
        pred_emb = model(x)
        return pred_emb[:, -loc_dim:]
    return wrapper

def check_time(rec):
    if ":"  in rec["time"]:
        hr = rec["time"].split(":")[0]
        mn = rec["time"].split(":")[1]
    elif "h" in rec["time"]:
        hr = rec["time"].split("h")[0]
        mn = rec["time"].split("h")[1]
    elif "?" in rec["time"]:
        return False
    else:
        raise ValueError("Unknown time format: ", rec["time"])
    
    hr = int(hr)
    mn = int(mn)
    # if hr >= 5 and hr <= 6 or (hr == 4 and mn >= 30) or (hr == 7 and mn <= 30):
    #     return True
    if hr == 5 or (hr == 6 and mn <= 30):
        return True
    return False


class DummyTqdm():
    def __init__(self):
        pass
    def tqdm(self, x, **kwargs):
        return x

location_encoder = LocationEncoder()
location_encoder = location_encoder.cuda()
location_encoder.eval()


def run_eval(all_spectrograms, model, metadata, metrics_list, use_true_location=False):

    counter = 0
    for path, file in tqdm.tqdm(all_spectrograms):
        counter += 1

        # if counter > 10: break

        fpath = os.path.join(path, file)

        spec_id = file.split(".")[0]
        
        if use_true_location:
            lat, lon = metadata[spec_id]["lat"], metadata[spec_id]["lng"]
            lat, lon = float(lat), float(lon)
            gt_loc = torch.Tensor([[lat, lon]]).cuda()
            predictions = location_encoder(gt_loc)

        else:
            frames_list, orig_frames = read_sound(fpath)
            softmax = torch.nn.Softmax(dim=-1)

            frames_all = torch.cat(frames_list, 0)
            frames_all = frames_all[:, 0, ...]
            m, st = frames_all.mean(), frames_all.std()
            m, st = 0.0, 1.0
            # m, st = 0.4, 1.0
            # m, st = -0.5, 2.0



            batch_size = 64
            num_batches = (len(frames_list) + batch_size - 1) // batch_size

            with torch.no_grad():
                predictions = [
                    model((torch.cat(frames_list[i*batch_size:min((i+1)*batch_size, len(frames_list))], 0).cuda() - m) / st)
                    for i in tqdm.tqdm(range(num_batches), desc="Predicting", leave=False)
                ]
                predictions = torch.cat(predictions, 0)


        lat, lon = metadata[spec_id]["latitude"], metadata[spec_id]["longitude"]
        lat, lon = float(lat)/90, float(lon)/180
        gt = torch.Tensor([[lat, lon]] * len(predictions))

        for metrics in metrics_list.values():
            metrics.update(
                predictions.cpu(), gt.cpu(), [fpath] * len(predictions)
            )

        
    return metrics_list




if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="mobilenet")
    parser.add_argument('--task_type', type=str, default="geoclip")
    parser.add_argument('--geo_resolution', type=float, default=0.0)
    parser.add_argument('--model_weight', type=str, default="")
    parser.add_argument('--ag_clip', action="store_true", default=False)
    parser.add_argument('--no_tqdm', action="store_true", default=False)
    parser.add_argument('--use_true_location', action="store_true", default=False)
    args = parser.parse_args()

    if args.no_tqdm:
        tqdm = DummyTqdm()

    # #xeno canto specific meta data
    df = pd.read_csv(config.XCDC_CSV)
    metadata = {str(df["audio_id"][i]): {k:df[k][i] for k in df.columns} for i in range(len(df["audio_id"]))}



    class DummyArgs():
        def __init__(self):
            self.bottleneck_dim = 128
            self.task_type = args.task_type


    geo_resolution = args.geo_resolution
    if args.task_type == "classification":
        resolutions = [name for name in args.model_weight.split("/") if name[:3] == "res"]
        if len(resolutions) > 0:
            geo_resolution = float(resolutions[0][3:])
        if geo_resolution < 1:
            output_dim = 122
        elif geo_resolution == 1.0:
            output_dim = 842
        else:
            output_dim = 5882
    else:
        output_dim= 512
    if args.ag_clip:
        output_dim = 5547 + 512
    model = models.get_model(
        args.model, 
        output_dim=output_dim, 
        pretrained=True, 
        dropout=0,
        args=DummyArgs()
    )
    if args.model_weight != "":
        weights = torch.load(args.model_weight, map_location="cpu", weights_only=True)
        model.load_state_dict(weights, strict=True)
    model.eval()
    model = model.cuda()
    if args.ag_clip:
        model = ag_clip_model_wrapper(model, loc_dim=512)

    
    all_spectrograms = []
    for path, dirs, files in os.walk(config.XCDC_SPECTROGRAM_DIR):
        if len(files) == 0: continue
        for file in files:
            if ".npy" not in file: continue
            spec_id = date = file.split(".")[0]
            all_spectrograms.append((path, file))


    galleries = {
        "iNatSounds train": None,
        "XCDC test": os.path.join(config.GALLERY_DIR, "gallery_xcdc.json"),
    }

    metrics_list = {
        gallery_name: utils.GeoLocalizationMetrics(
            args.task_type, 
            geo_resolution=geo_resolution, 
            geo_gallery="train",
            gallery_path_arg=gallery_path,
            save_dir = "./xeno_canto_v2/metrics"
        )
        for gallery_name, gallery_path in galleries.items()
    }
    metrics_list = run_eval(
        all_spectrograms, model, metadata,
        metrics_list,
        use_true_location=args.use_true_location
    )
    for k, m in metrics_list.items():
        print(k)
        print(m)
        print("="*20)


