from src import utils, config

import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import json
import h3
import os
import argparse
import random

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--eval_set', type=str, default="test")
args = parser.parse_args()


# set random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

def geodesic_distance(loc1, loc2, normalized=True):
    lat1, lon1 = loc1[..., 0] * 90, loc1[..., 1] * 180
    lat2, lon2 = loc2[..., 0] * 90, loc2[..., 1] * 180
    if normalized:
        r = 1/np.pi  # 2* Radius of Earth / circumference
    else:
        r = 6371
    phi1, phi2 = torch.deg2rad(lat1), torch.deg2rad(lat2)
    delta_phi, delta_lambda = torch.deg2rad(lat2-lat1), torch.deg2rad(lon2-lon1)
    a = torch.sin(delta_phi/2)**2 + torch.cos(phi1) * torch.cos(phi2) * torch.sin(delta_lambda/2)**2
    return 2 * r * torch.asin(torch.sqrt(a))



MODE =  args.eval_set


with open(config.TRAIN_JSON, "r") as f:
    train_data = json.load(f)
with open(config.VAL_JSON, "r") as f:
    val_data = json.load(f)
with open(config.TEST_JSON, "r") as f:
    test_data = json.load(f)
eval_data = val_data if MODE == "val" else test_data

resolutions = ["0.1", "1", "2"]
def date2yeardate(date):
    month = int(date.split("-")[1])
    day = int(date.split("-")[2][:2])
    return month, day

def date_distance(date1, date2):
    # (01, 20), (12, 20) 
    # (12, 20), (01, 20)
    month1, day1 = date2yeardate(date1)
    month2, day2 = date2yeardate(date2)

    rolled_date1 = 31 * month1 + day1
    rolled_date2 = 31 * month2 + day2

    return min(abs(rolled_date1 - rolled_date2), 365 - abs(rolled_date1 - rolled_date2))
    

train_res_geocell2audio = {}
for res in resolutions:
    train_res_geocell2audio[res] = {}
    for audio in train_data["audio"]:
        cell = h3.latlng_to_cell(audio["latitude"], audio["longitude"], float(res))
        if cell not in train_res_geocell2audio[res]:
            train_res_geocell2audio[res][cell] = []
        train_res_geocell2audio[res][cell].append(audio)

train_id2date = {
    audio["file_name"]:audio["date"] for audio in train_data["audio"]
}

lat_lon_scale = torch.Tensor([[90.0, 180.0]])

class2latlon = {}

res_class2geocell = {res:{} for res in resolutions}
geo_cells = {res:[] for res in resolutions}
for audio in tqdm.tqdm(train_data["audio"]):
    cl = audio["file_name"].split("/")[-2]
    lat_lon = [audio["latitude"], audio["longitude"]]
    if cl not in class2latlon:
        class2latlon[cl] = []
    class2latlon[cl].append(lat_lon)

    for res in resolutions:
        cell = h3.latlng_to_cell(lat_lon[0], lat_lon[1], float(res))
        geo_cells[res].append(cell)
        if cl not in res_class2geocell[res]:
            res_class2geocell[res][cl] = []
        res_class2geocell[res][cl].append(cell)
        
# Mean prediction
all_lat_lon = [l for c in class2latlon for l in class2latlon[c]]
all_lat_lon = np.array(all_lat_lon)
mean_lat_lon = np.mean(all_lat_lon, 0)
mean_lat_lon = torch.Tensor(mean_lat_lon).unsqueeze(0)


# Class-wise Mean prediction
class2latlon_np = {c:np.array(l) for c, l in class2latlon.items()}
class2mean_latlon = {
    c:torch.Tensor(np.mean(class2latlon_np[c], 0)).unsqueeze(0)  for c in tqdm.tqdm(class2latlon_np)
}

# majority predictions for all resolutions
all_classes = [c for res in resolutions for c in geo_cells[res]]
all_classes = list(set(all_classes))
classes2latlon = {
    c:torch.Tensor(list(h3.cell_to_latlng(c))).unsqueeze(0)
    for c in all_classes
}
res2latlon = {}
for res in resolutions:
    majority = max(set(geo_cells[res]), key=geo_cells[res].count)
    res2latlon[res] = classes2latlon[majority]


# Random predictions for all resolutions
res2cells = {}
for res in resolutions:
    res2cells[res] = list(set(geo_cells[res]))

# class-wise majority predictions for all resolutions
res_cls2latlon = {res:{} for res in resolutions}
for res in resolutions:
    for c in res_class2geocell[res]:
        ls = res_class2geocell[res][c]
        majority = max(set(ls), key=ls.count)
        res_cls2latlon[res][c] = classes2latlon[majority]






train_class2audio = {}
for a in train_data["audio"]:
    cl = a["file_name"].split("/")[-2]
    if cl not in train_class2audio:
        train_class2audio[cl] =[]
    train_class2audio[cl].append(a)


print("predictions made!")

save_root = "./trivial_baselines/"
all_metric_names = ["random", "mean", "class_mean", "class_pick", "random_pick"]
for name in ["majority", "random", "class_majority"]:
    for res in resolutions:
        all_metric_names.append("{}_res{}".format(name, res))

all_metric_names += [
    "species2date_best", "species2date_top5",
    "closest_species", "closest_species_top5"
]
for res in resolutions:
    all_metric_names.extend([
        "cell2species_random_res{}".format(res), "cell2species_best_res{}".format(res), "cell2species_top5_res{}".format(res),
    ])
all_metric_names.append("closest")

all_metrics = {n:utils.GeoLocalizationMetrics("lat_lon", 1, clip_only=True, save_dir = os.path.join(save_root, "run_{}".format(args.seed), MODE, n)) for n in all_metric_names}
# all_metrics["class_pick"].save_dir = "./class_pick_preds/"

# for audio in tqdm.tqdm(val_data["audio"]):
# for audio in tqdm.tqdm(test_data["audio"]):
# audio_i = 0
for audio in tqdm.tqdm(eval_data["audio"]):
    # audio_i += 1
    # if audio_i < 7000:
    #     continue
    label = [audio["latitude"], audio["longitude"]]
    label = torch.Tensor(label).unsqueeze(0) / lat_lon_scale

    # closest 
    all_distances = geodesic_distance(label, all_lat_lon / lat_lon_scale, normalized=False)
    best_match = all_lat_lon[np.argmin(all_distances)]
    all_metrics["closest"].update(
        best_match / lat_lon_scale, label, [os.path.basename(audio["file_name"])]
    )



    # Random lat/lon prediction
    random_pred = torch.rand(1, 2) * 2 - 1
    all_metrics["random"].update(
        random_pred, label, [os.path.basename(audio["file_name"])]
    )


    # Mean prediction
    all_metrics["mean"].update(
        mean_lat_lon / lat_lon_scale, label, [os.path.basename(audio["file_name"])]
    )

    # Class-wise Mean prediction
    cl = audio["file_name"].split("/")[-2]
    all_metrics["class_mean"].update(
        class2mean_latlon[cl] / lat_lon_scale, label, [os.path.basename(audio["file_name"])]
    )

    # majority prediction for all resolutions
    for res in resolutions:
        all_metrics["majority_res{}".format(res)].update(
            res2latlon[res] / lat_lon_scale, label, [os.path.basename(audio["file_name"])]
        )

    # Random prediction for all resolutions
    for res in resolutions:
        pred = classes2latlon[np.random.choice(res2cells[res])]
        all_metrics["random_res{}".format(res)].update(
            pred / lat_lon_scale, label, [os.path.basename(audio["file_name"])]
        )

    # class-wise majority prediction for all resolutions
    for res in resolutions:
        all_metrics["class_majority_res{}".format(res)].update(
            res_cls2latlon[res][cl] / lat_lon_scale, label, [os.path.basename(audio["file_name"])]
        )
    
    # class-wise randon picked prediction
    cur_class_latlon = class2latlon_np[cl]
    pred = cur_class_latlon[np.random.choice(cur_class_latlon.shape[0])]
    all_metrics["class_pick"].update(
        pred / lat_lon_scale, label, [os.path.basename(audio["file_name"])]
    )

    # randomly pick lat/lon from the train
    pred = all_lat_lon[np.random.choice(all_lat_lon.shape[0])]
    all_metrics["random_pick"].update(
        pred / lat_lon_scale, label, [os.path.basename(audio["file_name"])]
    )



    # 7 nov changes
    # same_class_samples = [audio for audio in train_data["audio"] if audio["file_name"].split("/")[-2] == cl]
    same_class_samples = train_class2audio[cl]


    # closest_species
    same_class_lat_lons = class2latlon_np[cl]
    same_class_lat_lons = torch.Tensor(same_class_lat_lons) / lat_lon_scale
    distances = geodesic_distance(label, same_class_lat_lons, normalized=False)
    best_match = same_class_lat_lons[torch.argmin(distances)]
    # print(best_match.shape, label.shape)
    best_match = best_match.unsqueeze(0)
    all_metrics["closest_species"].update(
        best_match, label, [os.path.basename(audio["file_name"])]
    )


    # closest_species_top5
    top5 = torch.topk(distances, 5, largest=False)
    top5_lat_lon = same_class_lat_lons[top5.indices]
    pred = top5_lat_lon[np.random.choice(5)].unsqueeze(0)
    all_metrics["closest_species_top5"].update(
        pred, label, [os.path.basename(audio["file_name"])]
    )


    # closest_date
    sample_dates = [train_id2date[a["file_name"]] for a in same_class_samples]

    distance_audio = [(date_distance(d, audio["date"]), i) for i, d in enumerate(sample_dates)]
    distance_audio = sorted(distance_audio, key=lambda x: x[0])

    # best match
    best_match = same_class_samples[distance_audio[0][1]]
    best_lat_lon = torch.Tensor([best_match["latitude"], best_match["longitude"]]).unsqueeze(0)
    all_metrics["species2date_best"].update(
        best_lat_lon / lat_lon_scale, label, [os.path.basename(audio["file_name"])]
    )

    # random from top 5
    top5 = distance_audio[:5]
    top5 = [same_class_samples[i] for _, i in top5]
    top5_lat_lon = torch.Tensor([[a["latitude"], a["longitude"]] for a in top5])
    pred = top5_lat_lon[np.random.choice(min(5, top5_lat_lon.shape[0]))]
    all_metrics["species2date_top5"].update(
        pred / lat_lon_scale, label, [os.path.basename(audio["file_name"])]
    )




    # test geo cell
    for res_i, res in enumerate(resolutions):
        geo_cell = h3.latlng_to_cell(audio["latitude"], audio["longitude"], float(res))

        if geo_cell not in train_res_geocell2audio[res]:
            # just pick random from any other cell
            cur_train_samples = same_class_samples
        else:
            cur_train_samples = train_res_geocell2audio[res][geo_cell]
        same_class = [audio for audio in cur_train_samples if audio["file_name"].split("/")[-2] == cl]
        if len(same_class) == 0:
            if res == "0.1":
                same_class = same_class_samples
            else:
                parent_res = resolutions[res_i - 1]
                parent_cell = h3.latlng_to_cell(audio["latitude"], audio["longitude"], float(parent_res))

                if parent_cell not in train_res_geocell2audio[parent_res]:
                    same_class = same_class_samples
                else:
                    cur_train_samples = train_res_geocell2audio[parent_res][parent_cell]
                    same_class = [audio for audio in cur_train_samples if audio["file_name"].split("/")[-2] == cl]
                    if len(same_class) == 0:
                        if res == "1":
                            same_class = same_class_samples
                        else:
                            grand_parent_res = resolutions[res_i - 2]
                            grand_parent_cell = h3.latlng_to_cell(audio["latitude"], audio["longitude"], float(grand_parent_res))

                            if grand_parent_cell not in train_res_geocell2audio[grand_parent_res]:
                                same_class = same_class_samples
                            else:
                                cur_train_samples = train_res_geocell2audio[grand_parent_res][grand_parent_cell]
                                same_class = [audio for audio in cur_train_samples if audio["file_name"].split("/")[-2] == cl]
                                if len(same_class) == 0:
                                    same_class = same_class_samples


        # pick train closest in date from the same cell
        dates = [train_id2date[a["file_name"]] for a in same_class]
        distance_audio = [(date_distance(d, audio["date"]), i) for i, d in enumerate(dates)]
        distance_audio = sorted(distance_audio, key=lambda x: x[0])

        # random
        pred = torch.Tensor([same_class[np.random.choice(len(same_class))]["latitude"], same_class[np.random.choice(len(same_class))]["longitude"]]).unsqueeze(0)
        all_metrics["cell2species_random_res{}".format(res)].update(
            pred / lat_lon_scale, label, [os.path.basename(audio["file_name"])]
        )

        # best match
        best_match = same_class[distance_audio[0][1]]
        best_lat_lon = torch.Tensor([best_match["latitude"], best_match["longitude"]]).unsqueeze(0)
        all_metrics["cell2species_best_res{}".format(res)].update(
            best_lat_lon / lat_lon_scale, label, [os.path.basename(audio["file_name"])]
        )

        # random from top 5
        top5 = distance_audio[:5]
        top5 = [same_class[i] for _, i in top5]
        top5_lat_lon = torch.Tensor([[a["latitude"], a["longitude"]] for a in top5])
        pred = top5_lat_lon[np.random.choice(min(5, top5_lat_lon.shape[0]))]
        all_metrics["cell2species_top5_res{}".format(res)].update(
            pred / lat_lon_scale, label, [os.path.basename(audio["file_name"])]
        )

        
    # class-wise randon picked prediction
    pred_cl = filename2predcls[audio["file_name"]]
    cur_class_latlon = class2latlon_np[pred_cl]
    pred = cur_class_latlon[np.random.choice(cur_class_latlon.shape[0])]
    all_metrics["pred_class_pick"].update(
        pred / lat_lon_scale, label, [os.path.basename(audio["file_name"])]
    )

    # "pred_species2date_best", "pred_species2date_top5",
    # "pred_closest_species", "pred_closest_species_top5"

    # closest_species
    same_class_lat_lons = class2latlon_np[pred_cl]
    same_class_lat_lons = torch.Tensor(same_class_lat_lons) / lat_lon_scale
    distances = geodesic_distance(label, same_class_lat_lons, normalized=False)
    # print(distances.shape, same_class_lat_lons.shape, pred_cl in present_classes_val_cl)
    best_match = same_class_lat_lons[torch.argmin(distances)]
    # print(best_match.shape, label.shape)
    best_match = best_match.unsqueeze(0)
    all_metrics["pred_closest_species"].update(
        best_match, label, [os.path.basename(audio["file_name"])]
    )


    # closest_species_top5
    top5 = torch.topk(distances, min(distances.shape[0], 5), largest=False)
    top5_lat_lon = same_class_lat_lons[top5.indices]
    pred = top5_lat_lon[np.random.choice(min(distances.shape[0], 5))].unsqueeze(0)
    all_metrics["pred_closest_species_top5"].update(
        pred, label, [os.path.basename(audio["file_name"])]
    )


    # closest_date
    pred_class_samples = train_class2audio[pred_cl]
    sample_dates = [train_id2date[a["file_name"]] for a in pred_class_samples]

    distance_audio = [(date_distance(d, audio["date"]), i) for i, d in enumerate(sample_dates)]
    distance_audio = sorted(distance_audio, key=lambda x: x[0])

    # best match
    best_match = pred_class_samples[distance_audio[0][1]]
    best_lat_lon = torch.Tensor([best_match["latitude"], best_match["longitude"]]).unsqueeze(0)
    all_metrics["pred_species2date_best"].update(
        best_lat_lon / lat_lon_scale, label, [os.path.basename(audio["file_name"])]
    )

    # random from top 5
    top5 = distance_audio[:min(5, len(distance_audio))]
    top5 = [pred_class_samples[i] for _, i in top5]
    top5_lat_lon = torch.Tensor([[a["latitude"], a["longitude"]] for a in top5])
    pred = top5_lat_lon[np.random.choice(min(5, top5_lat_lon.shape[0]))]
    all_metrics["pred_species2date_top5"].update(
        pred / lat_lon_scale, label, [os.path.basename(audio["file_name"])]
    )



print("Using Random Seed {}".format(args.seed))
for name in all_metrics:
    print("="*20)
    print(name)
    print(all_metrics[name])