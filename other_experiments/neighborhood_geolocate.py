
import torch
import tqdm
import h3
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

from src import utils, config


parser = argparse.ArgumentParser()
parser.add_argument('--pred_json', type=str, default="")
args = parser.parse_args()


def get_performance(pred_dict, name2latlon_torch, res=None):
    metrics = utils.GeoLocalizationMetrics(
        "geoclip", 
        geo_resolution=0.1, 
        geo_gallery="train",
        save_dir="hexa_predictions/" + str(res) if res is not None else "normal",
    )
        
    for name in pred_dict:
        pred_embed = pred_dict[name]
        metrics.update(
            pred_embed, name2latlon_torch[name], [name]
        )
    return metrics

def date2days(date):
    date = date.split("-")
    year = int(date[0])
    month = int(date[1])
    day = int(date[2])
    return  (month - 1) * 30 + day

avg_pred_embed_path = args.pred_json
avg_pred_embed_dict = utils.read_json(avg_pred_embed_path)
avg_pred_embed_dict = {
    name:torch.tensor(avg_pred_embed_dict[name])
    for name in avg_pred_embed_dict
}

test_data = utils.read_json(config.TEST_JSON)
name2date = {
    audio["file_name"].split(".")[0]:date2days(audio["date"].split("T")[0])
    for audio in test_data["audio"]
}
name2latlon = {
    audio["file_name"].split(".")[0]:(audio["latitude"], audio["longitude"])
    for audio in test_data["audio"]
}
name2latlon_torch = {
    name:torch.tensor([latlon[0]/90, latlon[1]/180]).unsqueeze(0)
    for name, latlon in name2latlon.items()
}
name2class = {
    audio["file_name"].split(".")[0]:os.path.basename(os.path.dirname(audio["file_name"]))
    for audio in test_data["audio"]
}

date_groups = {
    "week": 7,
    "month": 30,
    # "season": 90,
}


hexa_resolutions = [5, 6]


hexa_groups = {}
for res in hexa_resolutions:
    hexa_groups[res] = {}
    for name in name2latlon:
        lat, lon = name2latlon[name]
        hexa_id = h3.latlng_to_cell(lat, lon, res)
        if hexa_id not in hexa_groups[res]:
            hexa_groups[res][hexa_id] = []
        hexa_groups[res][hexa_id].append(name)

for res in hexa_resolutions:
    for date_name, days in date_groups.items():
        key = "{}_{}".format(res, date_name)
        hexa_groups[key] = {}
        for name in name2latlon:
            lat, lon = name2latlon[name]
            hexa_id = h3.latlng_to_cell(lat, lon, res)
            cur_key = "{}_{}".format(hexa_id, name2date[name] // days)
            if cur_key not in hexa_groups[key]:
                hexa_groups[key][cur_key] = []
            hexa_groups[key][cur_key].append(name)
        # print(sum(
        #     len(hexa_groups[key][cur_key])
        #     for cur_key in hexa_groups[key]
        # ))

    

for res in hexa_groups:
    print("="*20)
    print("Geo Resolution {} grouped".format(res))
    sizes = [len(hexa_groups[res][hexa_id]) for hexa_id in hexa_groups[res]]
    sizes = np.array(sizes)
    print("Numbers: 99th Percentile {:.2f} Average {:.2f} Median {:.2f} Min {:.2f} Max {:.2f}".format(
        np.percentile(sizes, 99),
        np.mean(sizes), np.median(sizes),
        np.min(sizes), np.max(sizes)
    ))
    plt.plot(sorted(sizes))
    plt.yscale("log")
    plt.title("Number of samples in each hexa (resolution {})".format(res))
    plt.xlabel("Hexa ID")
    plt.ylabel("Number of samples")
    plt.savefig("hexa_sizes/hexa_sizes_{}.png".format(res))
    plt.close()

    unique_classes = [
        len(set([name2class[name] for name in hexa_groups[res][hexa_id]]))
        for hexa_id in hexa_groups[res]
    ]
    unique_classes = np.array(unique_classes)
    print("Classes: 99th Percentile {:.2f} Average {:.2f} Median {:.2f} Min {:.2f} Max {:.2f}".format(
        np.percentile(unique_classes, 99),
        np.mean(unique_classes), np.median(unique_classes),
        np.min(unique_classes), np.max(unique_classes)
    ))
exit()

grouped_avg_predictions = {}
for res in hexa_groups:
    grouped_avg_predictions[res] = {}
    for hexa_id in hexa_groups[res]:
        all_names = hexa_groups[res][hexa_id]
        all_preds = []
        for name in all_names:
            all_preds.append(avg_pred_embed_dict[name])
        all_preds = torch.stack(all_preds)
        cur_avg_pred_embed = torch.mean(all_preds, dim=0)
        for name in all_names:
            grouped_avg_predictions[res][name] = cur_avg_pred_embed


# normal_metrics = get_performance(avg_pred_embed_dict, name2latlon_torch)
# print("Normal metrics")
# print(normal_metrics)

for res in hexa_groups:
    print("Geo Resolution {} grouped".format(res))
    metrics = get_performance(grouped_avg_predictions[res], name2latlon_torch, res)
    print(metrics)

