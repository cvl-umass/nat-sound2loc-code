from transformers import PretrainedConfig
from rshf.taxabind import TaxaBind
import torchaudio
import torch
import torch.nn.functional as F
import utils
import tqdm
import os
from multiprocessing import Process
import argparse





def process_audio_embeds(rank, n_threads):

    config = PretrainedConfig.from_pretrained("MVRL/taxabind-config")
    # config.audio_encoder = "laion/clap-htsat-fused"
    taxabind = TaxaBind(config)
    audio_encoder = taxabind.get_audio_encoder()
    audio_encoder.eval()
    audio_processor = taxabind.get_audio_processor()


    os.makedirs(AUDIO_EMBED_SAVE_PATH, exist_ok=True)

    dataset = utils.read_json(dataset_path)

    cur_size = (len(dataset["audio"]) + n_threads - 1) // n_threads
    start = rank * cur_size
    end = min((rank + 1) * cur_size, len(dataset["audio"]))

    for i in tqdm.tqdm(range(start, end), leave=False):
        audio_dict = dataset["audio"][i]
        fname = audio_dict["file_name"]
        lat, lon = audio_dict["latitude"], audio_dict["longitude"]

        audio, sr = torchaudio.load(os.path.join(SOUND_ROOT, fname))
        audio = audio_processor(audio, sr)
        with torch.no_grad():
            audio_embeds = audio_encoder(**audio).audio_embeds
            audio_embeds = F.normalize(audio_embeds, dim=1)
        torch.save(audio_embeds, os.path.join(AUDIO_EMBED_SAVE_PATH, f"{i}.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()


    SOUND_ROOT = "/scratch3/workspace/mchasmai_umass_edu-inat_sound/data/inatsounds_release"
    DATASET_JSON = "/scratch3/workspace/mchasmai_umass_edu-inat_sound/data/inatsounds_release/{}.json"
    SPLIT = args.split
    dataset_path = DATASET_JSON.format(SPLIT)
    AUDIO_EMBED_SAVE_PATH = "/scratch3/workspace/mchasmai_umass_edu-inat_sound/data/taxabind_audio_embeds/{}".format(SPLIT)




    n_threads = 1

    processes = []
    for rank in range(n_threads):
        p = Process(target=process_audio_embeds, args=(rank, n_threads))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    
