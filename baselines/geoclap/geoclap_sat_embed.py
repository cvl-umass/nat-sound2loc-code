# from geoclap.models.SATMAE import SatMAE
from geoclap.train import GeoCLAP
import torch
from torchvision import transforms
from torchvision.io import read_image
import json
import tqdm
import os
import numpy as np

def read_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def write_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f)



ckpt_path = "./geoclap/checkpoints/geoclap_soundingEarth_best_model.ckpt"

pretrained_ckpt = torch.load(ckpt_path)
hparams = pretrained_ckpt['hyper_parameters']
print(hparams)
pretrained_weights = pretrained_ckpt['state_dict']
model = GeoCLAP(hparams).to("cuda")
model.load_state_dict(pretrained_weights)
geoclap = model.eval()


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
sat_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
])

split = "train"
SAT_DIR = "/scratch3/workspace/mchasmai_umass_edu-inat_sound/data/satellite_images/"
JSON_FORMAT = "/scratch3/workspace/mchasmai_umass_edu-inat_sound/data/inatsounds_release/{}.json"
dataset = read_json(JSON_FORMAT.format(split))
SAVE_DIR = "/scratch3/workspace/mchasmai_umass_edu-inat_sound/data/galleries/geoclap"
os.makedirs(SAVE_DIR, exist_ok=True)



geoclap_gallery = {}
gallery_save_path = os.path.join(SAVE_DIR, "gallery_{}.json".format(split))

for i, audio_dict in enumerate(tqdm.tqdm(dataset["audio"])):
    fname = audio_dict["file_name"]
    lat, lon = audio_dict["latitude"], audio_dict["longitude"]

    sat_path = os.path.join(SAT_DIR, fname.replace("/", "_").replace(".wav", ".png"))   
    # print(sat_path)
    sat_img = read_image(sat_path, mode="RGB")
    # print(sat_img.shape)
    sat_img = np.array(torch.permute(sat_img, [1, 2, 0]))

    # print(sat_img.shape)
    sat_img = sat_transforms(sat_img).unsqueeze(0).to("cuda")

    with torch.no_grad():
        sat_embeds = geoclap.sat_encoder(sat_img)
    sat_embeds = sat_embeds.detach().cpu()

    geoclap_gallery["{}_{}".format(lat, lon)] = sat_embeds.view(-1).numpy().tolist()

    # print(fname, sat_embeds.shape)

    if i%1000 == 0:
        write_json(geoclap_gallery, gallery_save_path)

write_json(geoclap_gallery, gallery_save_path)

