import urllib
import os

import urllib.request
from PIL import Image
import os
import math
import tqdm
from multiprocessing import Process

from src import config

import json
def read_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def save_image(api_key, lat, lon, filename, temp_name="temp"):

    url = "https://maps.googleapis.com/maps/api/staticmap?center={:.7f},{:.7f}&zoom=20&size=512x512&scale=1&maptype=satellite&key={}".format(lat, lon, api_key)
    temp_file = temp_name
    urllib.request.urlretrieve(url, temp_file)
    im = Image.open(temp_file)
    im.save(filename)

def process_sat_images(dataset, left_indices, args.save_dir, api_key, rank=0):
    for i in tqdm.tqdm(left_indices, leave=False):
        audio_dict = dataset["audio"][i]
        fname = audio_dict["file_name"]
        lat, lon = audio_dict["latitude"], audio_dict["longitude"]

        try:
            save_image(api_key, lat, lon, os.path.join(args.save_dir, fname.replace("/", "_").replace(".wav", ".png")), temp_name="temp_{}".format(rank))
        except KeyboardInterrupt:
            exit()
        except Exception as e:
            print(f"Error saving {fname}", e)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="./baselines/geoclap/satellite_images")
    parser.add_argument('--google_api_key', type=str, default="")
    args = parser.parse_args()

    split = "train"
    JSON_FORMAT = "/scratch3/workspace/mchasmai_umass_edu-inat_sound/data/inatsounds_release/{}.json"
    os.makedirs(args.save_dir, exist_ok=True)

    dataset = read_json(config.JSON_DICT[split])

    # save_image(29.8066414, -95.7655343, "high_resolution_image_2.png")

    left_indices = [
        i for i, audio_dict in enumerate(dataset["audio"])
        if os.path.exists(os.path.join(args.save_dir, audio_dict["file_name"].replace("/", "_").replace(".wav", ".png")))==False
    ]

    # process_sat_images(dataset, left_indices, args.save_dir)


    n_threads = 16

    thread_size = (len(left_indices) + n_threads - 1) // n_threads

    processes = []
    for rank in range(n_threads):
        cur_indices = left_indices[rank*thread_size:min((rank+1)*thread_size, len(left_indices))]
        p = Process(target=process_sat_images, args=(dataset, cur_indices, args.save_dir, args.google_api_key, rank))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    


if __name__ == '__main__': 
    main()



