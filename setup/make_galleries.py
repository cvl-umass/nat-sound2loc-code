from geoclip import LocationEncoder
import json
import torch
import numpy as np
import cv2
import os
from satclip.load import get_satclip
import pandas as pd

from src import config


def location_list2gallery(location_encoder, locations, double=False):
    all_lat_lon = torch.Tensor(locations)
    if double:
        all_lat_lon = all_lat_lon.double()
    all_lat_lon_embed = location_encoder(all_lat_lon)
    all_lat_lon_embed = all_lat_lon_embed.detach().numpy()
    all_lat_lon_embed = all_lat_lon_embed.tolist()

    gallery = {
        "_".join([str(l) for l in loc]): embed
        for loc, embed in zip(locations, all_lat_lon_embed)
    }
    return gallery


def make_geoclip_gallery(locations, vis_path=None, point_size=2):
    print("Making Gallery of size", len(locations))

    if vis_path is not None:
        cur_map = world_map.copy()
        loc_np = np.array(locations)
        xs = np.int32((loc_np[:, 1] + 180) * world_map.shape[1] / 360)
        ys = np.int32((90 - loc_np[:, 0])* world_map.shape[0] / 180)
        coordinates = np.stack([xs, ys], -1) # N x 2
        # unique coordinates
        xy = np.unique(coordinates, axis=0)

        for x, y in xy.tolist():
            cv2.circle(cur_map, (x, y), point_size, (255, 0, 0), -1)
        cv2.imwrite(vis_path, cur_map)

    location_encoder = LocationEncoder()
    location_encoder = location_encoder.cuda()
    location_encoder.eval()


    all_lat_lon = torch.Tensor(locations).cuda()
    all_lat_lon_embed = location_encoder(all_lat_lon)
    all_lat_lon_embed = all_lat_lon_embed.cpu().detach().numpy()
    all_lat_lon_embed = all_lat_lon_embed.tolist()


    gallery = {
        "_".join([str(l) for l in loc]): embed
        for loc, embed in zip(locations, all_lat_lon_embed)
    }

    return gallery


  

if __name__ == "__main__":

    gallery_dir = config.GALLERY_DIR
    os.makedirs(gallery_dir, exist_ok=True)



    # make gallery from all training locations
    with open(config.TRAIN_JSON, "r") as f:
        train_data = json.load(f)
    all_lat_lon_list = [
        [a["latitude"], a["longitude"]]
        for a in train_data["audio"]
    ]

    train_gallery = make_geoclip_gallery(all_lat_lon_list)
    with open(os.path.join(gallery_dir, "gallery_train.json"), "w") as f:
        json.dump(train_gallery, f)
    print("train gallery made!")


    # make gallery from all training locations
    with open(config.VAL_JSON, "r") as f:
        train_data = json.load(f)
    all_lat_lon_list = [
        [a["latitude"], a["longitude"]]
        for a in train_data["audio"]
    ]

    val_gallery = make_geoclip_gallery(all_lat_lon_list)
    with open(os.path.join(gallery_dir, "gallery_val.json"), "w") as f:
        json.dump(val_gallery, f)
    print("val gallery made!")







    # make gallery from all land locations

    ocean = np.load(config.OCEAN_MASK)
    print(ocean.shape)
    res = 5

    all_lat_lon_list = []
    for i in range(ocean.shape[1]):
        for j in range(ocean.shape[0]):
            if i % res != 0 or j % res != 0: continue
            lat = 90 - 180 * j / ocean.shape[0]
            lon = 360 * i / ocean.shape[1] - 180
            if ocean[j, i] == 1:
                all_lat_lon_list.append([lat, lon])

    land_gallery = make_geoclip_gallery(all_lat_lon_list)
    with open(os.path.join(gallery_dir, "gallery_land.json"), "w") as f:
        json.dump(land_gallery, f)

    print("land gallery made!")




    location_encoder = get_satclip("../satclip-vit16-l40.ckpt", device="cpu")
    location_encoder.eval()
    def satclip_encode(x):
        return location_encoder(torch.flip(x, [-1]))
    with open(config.TRAIN_JSON, "r") as f:
        train_data = json.load(f)
    all_lat_lon_list = [
        [a["latitude"], a["longitude"]]
        for a in train_data["audio"]
    ]

    train_gallery = location_list2gallery(satclip_encode, all_lat_lon_list, double=True)
    with open(os.path.join(gallery_dir, "gallery_satclip.json"), "w") as f:
        json.dump(train_gallery, f)
    print("satclip train gallery made!")
    








    df = pd.read_csv(config.XCDC_CSV)
    all_locations = [
        (float(df["latitude"][i]), float(df["longitude"][i]) )
        for i in range(len(df["audio_id"]))
    ]
    gallery = make_geoclip_gallery(all_locations)
    with open(os.path.join(gallery_dir, "gallery_xcdc.json"), "w") as f:
        json.dump(gallery, f)