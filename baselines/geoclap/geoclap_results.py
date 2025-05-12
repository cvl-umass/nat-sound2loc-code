# from geoclap.models.SATMAE import SatMAE
from geoclap.train import GeoCLAP
from transformers import ClapProcessor
import torch
from torchvision import transforms
from torchvision.io import read_image
import json
import tqdm
import os
import numpy as np
import utils
import torch.nn.functional as F
import torchaudio

from src import config

# from transformers import pipeline

# audio_classifier = pipeline(task="zero-shot-audio-classification", model="laion/clap-htsat-fused")


def read_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def write_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f)


class GeoCLAPMetrics(utils.GeoLocalizationMetrics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        print("getting location embeddings!")
        gallery_path = os.path.join(config.GALLERY_DIR, "{}/gallery_{}.json".format("geoclap", "train"))
        # gallery_path = gallery_path.replace("sinr", "geoclip")
        try:
            loc_embeds = read_json(gallery_path)
        except FileNotFoundError:
            raise ValueError("Could not find gallery in {}. Please ensure you have run ./misc/make_gallery.py".format(gallery_path))
        # loc_embeds format: {"lat_lon": [0.12, 0.7 ... 512 dim]}
        lat_lon = loc_embeds.keys()
        lat_lon = [
            [float(l.split("_")[0]), float(l.split("_")[1])]
            for l in lat_lon
        ]
        self.locations = torch.Tensor(lat_lon)
        self.loc_embeds = torch.Tensor(list(loc_embeds.values()))
        self.loc_embeds = F.normalize(self.loc_embeds, dim=1)


metrics = GeoCLAPMetrics(
    task_type="geoclip",
    geo_gallery="train",
    geo_resolution = 1,
    clip_only=True,
    save_dir="baseline_metrics/geoclap"
)


ckpt_path = "./geoclap/checkpoints/geoclap_soundingEarth_best_model.ckpt"

pretrained_ckpt = torch.load(ckpt_path)
hparams = pretrained_ckpt['hyper_parameters']
print(hparams)
pretrained_weights = pretrained_ckpt['state_dict']
model = GeoCLAP(hparams).to("cuda")
model.load_state_dict(pretrained_weights)
geoclap = model.eval()


processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
SAMPLE_RATE = 48000

def get_audio_clap(path_to_audio,padding="repeatpad",truncation="fusion"):
    track, sr = torchaudio.load(path_to_audio,format="mp3")
    track = track.mean(axis=0)
    track = torchaudio.functional.resample(track, orig_freq=sr, new_freq=SAMPLE_RATE)
    output = processor(audios=track, sampling_rate=SAMPLE_RATE, max_length_s=10, return_tensors="pt",padding=padding,truncation=truncation)
    # print(output['input_features'].shape)
    # output['input_features'] = output['input_features'][0,:,:,:]
    return output




split = "test"
SOUND_ROOT = "/scratch3/workspace/mchasmai_umass_edu-inat_sound/data/inatsounds_release"
JSON_FORMAT = "/scratch3/workspace/mchasmai_umass_edu-inat_sound/data/inatsounds_release/{}.json"
dataset = read_json(JSON_FORMAT.format(split))




for i, audio_dict in enumerate(tqdm.tqdm(dataset["audio"])):
    if i < 10: continue
    fname = audio_dict["file_name"]
    lat, lon = audio_dict["latitude"], audio_dict["longitude"]

    audio_path = os.path.join(SOUND_ROOT, fname)
    audio = get_audio_clap(path_to_audio=audio_path)

    audio = audio.to("cuda")
    with torch.no_grad():
        audio_embeds = geoclap.audio_encoder(audio)
    audio_embeds = audio_embeds.detach().cpu()

    metrics.update(
        audio_embeds.cpu(),
        torch.Tensor([[lat/90, lon/180]]),
        [fname]
    )

    if (i+1) % 1000 == 0:
        metrics_to_print = str(metrics)
        with open("geoclap_{}_results.txt".format(split), "a") as f:
            f.write(metrics_to_print)
            f.write("\n\n")


print(metrics)
with open("geoclap_{}_results.txt".format(split), "a") as f:
            f.write(metrics_to_print)
            f.write("\n\n")



