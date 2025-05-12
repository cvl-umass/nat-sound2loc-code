from transformers import PretrainedConfig
from rshf.taxabind import TaxaBind
import torchaudio
import torch
import torch.nn.functional as F
import utils
import tqdm
import os
import numpy as np

config = PretrainedConfig.from_pretrained("MVRL/taxabind-config")
# config.audio_encoder = "laion/clap-htsat-fused"
taxabind = TaxaBind(config)

print(config)

location_encoder = taxabind.get_location_encoder()


audio_encoder = taxabind.get_audio_encoder()
audio_encoder.eval()
audio_encoder = audio_encoder.cuda()
audio_processor = taxabind.get_audio_processor()


# audio, sr = torchaudio.load("/scratch3/workspace/mchasmai_umass_edu-inat_sound/data/inatsounds_release/test/05561_Animalia_Chordata_Reptilia_Squamata_Viperidae_Crotalus_oreganus/d0fafae6-3fae-4db5-bf2f-658a0efecbab.wav")
# audio_embed = audio_encoder(audio)
# audio = audio_processor(audio, sr)
# audio_embed = audio_encoder(**audio)
# audio_embed = audio_embed.audio_embeds
# print(audio_embed.shape)


loc_embeds = location_encoder(
    torch.Tensor(
        [
            [20, 20],
            [70, 160],
            [-70, -160],
        ]
    )
)
# print(loc_embeds)
print(loc_embeds.shape)




class TaxaBindMetrics(utils.GeoLocalizationMetrics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.location_encoder = taxabind.get_location_encoder()
        self.location_encoder = self.location_encoder.eval()
        self.location_encoder = self.location_encoder.cuda()

        print("getting location embeddings!")
        print(self.locations.min(), self.locations.max())
        with torch.no_grad():
            self.loc_embeds = self.location_encoder(self.locations.cuda())
            self.loc_embeds = F.normalize(self.loc_embeds, dim=1) 
        self.loc_embeds = self.loc_embeds.cpu()
        print(self.loc_embeds.shape)


metrics = TaxaBindMetrics(
    task_type="geoclip",
    geo_gallery="train",
    geo_resolution = 1,
    clip_only=True,
    save_dir="baseline_metrics/taxabind",
)

RESOLUTION = 1
SOUND_ROOT = "/scratch3/workspace/mchasmai_umass_edu-inat_sound/data/inatsounds_release"
DATASET_JSON = "/scratch3/workspace/mchasmai_umass_edu-inat_sound/data/inatsounds_release/{}.json"
DEVICE = "cuda"

SPLIT = "test"
val_set = utils.read_json(DATASET_JSON.format(SPLIT))

# for i, audio_dict in enumerate(tqdm.tqdm(val_set["audio"])):
# for i, audio_dict in enumerate(val_set["audio"]):
indices = list(range(len(val_set["audio"])))
np.random.shuffle(indices)
for i in tqdm.tqdm(indices):
    audio_dict = val_set["audio"][i]
    fname = audio_dict["file_name"]
    lat, lon = audio_dict["latitude"], audio_dict["longitude"]

    audio, sr = torchaudio.load(os.path.join(SOUND_ROOT, fname))
    audio = audio_processor(audio, sr)
    # max_length_s=10, return_tensors="pt",padding="repeatpad",truncation="fusion"
    audio = audio.to(DEVICE)
    with torch.no_grad():
        audio_embeds = audio_encoder(**audio).audio_embeds
    
    metrics.update(
        audio_embeds.cpu(),
        torch.Tensor([[lat/90, lon/180]]),
        [fname]
    )

    # if (i+1) % 1000 == 0:
    #     metrics_to_print = str(metrics)
    #     with open("taxabind_{}_results_rnd.txt".format(SPLIT), "a") as f:
    #         f.write(metrics_to_print)
    #         f.write("\n\n")

print(metrics)