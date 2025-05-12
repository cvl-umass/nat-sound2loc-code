import utils
import numpy as np
import cv2
import h3
import matplotlib.pyplot as plt
plasma = plt.get_cmap('plasma')
import os
import matplotlib
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import LogLocator
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

ocean = np.load("/scratch3/workspace/mchasmai_umass_edu-inat_sound/data/geo_model/data/masks/ocean_mask.npy")
ocean = ocean
h, w = ocean.shape
ocean = cv2.cvtColor(ocean, cv2.COLOR_GRAY2RGB)
print(ocean.shape)

# resolutions = [0, 1, 2]
resolutions = [2]

eval_data = utils.read_json("/scratch3/workspace/mchasmai_umass_edu-inat_sound/data/inatsounds_release/test.json")
img_name2loc = {
    a["file_name"].split(".")[0]: (a["latitude"], a["longitude"])
    for a in eval_data["audio"]
}
img_just_name2loc = {
    a["file_name"].split(".")[0].split("/")[-1]: (a["latitude"], a["longitude"])
    for a in eval_data["audio"]
}

def errors2score(error_dict):

    max_error, min_error = max(error_dict.values()), min(error_dict.values())
    scores = {cell: np.log(error) for cell, error in error_dict.items()}
    max_score, min_score = max(scores.values()), min(scores.values())

    # max_score = np.log(20000)
    # min_score = np.log(0.2)

    if max_score == min_score:
        return {cell: 1.0 for cell in scores}
    scores = {cell: ((score - min_score)/(max_score - min_score)) for cell, score in scores.items()}


    # ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
    # tick_labels = [min_error + (max_error - min_error) * tick for tick in ticks]
    # tick_labels = [np.exp(l) for l in tick_labels]
    # tick_labels = ["{}km".format(int(l)) for l in tick_labels]

    tick_labels = [1, 5, 25, 100, 1000, 10000]
    tick_labels = [l for l in tick_labels if l >= min_error and l <= max_error]
    ticks = [np.log(l) for l in tick_labels]
    ticks = [(l - min_score)/(max_score - min_score) for l in ticks]

    tick_labels = ["{}km".format(int(l) if l <= 100 else "10$^{}$".format(int(np.log10(l)))) for l in tick_labels]

    return scores, ticks, tick_labels


pixel2cell_res = {}
cell_res2indx = {}
for res in resolutions:
    # pixel2cell_res[res] = {}
    cell_res2indx[res] = {}
    for i in range(h):
        for j in range(w):
            lat = (0.5 - i/h) * 180
            lon = (j/w - 0.5) * 360
            cell = h3.latlng_to_cell(lat, lon, res)
            # pixel2cell_res[res][(i, j)] = cell
            if cell not in cell_res2indx[res]:
                cell_res2indx[res][cell] = []
            cell_res2indx[res][cell].append((i, j))


for res in resolutions:
    for lat, lon in img_name2loc.values():
        cell = h3.latlng_to_cell(lat, lon, 5)
        i, j = h * (0.5 - lat/180), w * (lon/360 + 0.5)
        i, j = int(i), int(j)
        if cell not in cell_res2indx[res]:
            cell_res2indx[res][cell] = []
        cell_res2indx[res][cell].append((i, j))

for res in resolutions:
    for cell in cell_res2indx[res]:
        cell_res2indx[res][cell] = np.array(list(set(cell_res2indx[res][cell])))



def plot_performance_map(preds_path, res, metric="average", save_dir="./performance_maps/", title_concat="", ret_map=False):
    os.makedirs(save_dir, exist_ok=True)
    preds = utils.read_json(preds_path)
    cell2errors = {}
    for img_name in preds:
        img_name_k = img_name
        if ".wav" in img_name:
            img_name_k = img_name.replace(".wav", "")
        if "/" in img_name:
            lat, lon = img_name2loc[img_name_k]
        else:
            lat, lon = img_just_name2loc[img_name_k]
        cell = h3.latlng_to_cell(lat, lon, res)
        dist = preds[img_name][metric][0]

        if cell not in cell2errors:
            cell2errors[cell] = []
        cell2errors[cell].append(dist)

    cell2errors = {cell: np.array(errors) for cell, errors in cell2errors.items()}

    cell2avg = {cell: np.mean(cell2errors[cell]) for cell in cell2errors}
    cell2median = {cell: np.median(cell2errors[cell]) for cell in cell2errors}


    cell2median_score, median_ticks, median_tick_labels = errors2score(cell2median)

    world_map_median = np.ones((h, w, 3)) * 255 * plasma(0.0)[:3]
    world_map_median = np.ones((h, w, 3)) * 200

    for cell in cell2median_score:
        color = np.array(plasma(cell2median_score[cell])[:3]) * 255
        world_map_median[cell_res2indx[res][cell][:, 0], cell_res2indx[res][cell][:, 1]] = color
    all_distances_median = [cell2median_score[cell] for cell in cell2median_score]

    world_map_median = cv2.cvtColor(world_map_median.astype(np.uint8), cv2.COLOR_RGB2BGR)
    world_map_median = np.where(ocean == 0, 255, world_map_median)




    plt.figure(figsize=(5.5, 5.5))
    sm = ScalarMappable(cmap='plasma')
    # sm.set_array(all_distances_median)
    # sm.set_array(np.arange(np.log(0.2), np.log(20000), 0.01))
    sm.set_array(np.arange(0, 1, 0.01))
    # fraction=0.02, 
    cbar = plt.colorbar(sm, ax=plt.gca(), orientation='vertical', location='right', pad=0.03, aspect=12, shrink=0.3)
    # cbar.ax.set_xticks(median_ticks, labels=median_tick_labels)
    cbar.ax.set_yticks(median_ticks, labels=median_tick_labels)
    cbar.ax.xaxis.set_ticks_position('bottom')
    # cbar.ax.yaxis.set_ticks_position('right')
    # cbar.ax.set_yticklabels(median_tick_labels)
    plt.imshow(cv2.cvtColor(world_map_median[:800, ...], cv2.COLOR_BGR2RGB))
    plt.axis("off")
    # plt.title("Median Geoloc Error " + title_concat, pad=-0.1)
    plt.gca().set_title("Median Geolocation Error " + title_concat, pad=-20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"median_{res}.png"), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.savefig(os.path.join(save_dir, f"median_{res}.pdf"), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    # cv2.imwrite(os.path.join(save_dir, f"avg_{res}.png"), world_map_avg)
    # cv2.imwrite(os.path.join(save_dir, f"median_{res}.png"), world_map_median)







save_dir = "./performance_maps/"
for res in resolutions:
    
    plot_performance_map(
        "/scratch3/workspace/mchasmai_umass_edu-inat_sound/output_logs/sound2loc/main_neurips/inat_pre/bot128/audio_geoclip/gallery_train/clip_scale/bce_0.01/pretrained/sound_aug/mobilenet/nesterov_b128_lr0.05_wd1e-05/train/2/preds/test/epoch_49/img2scale_distances.json",
        res=res,
        metric="average",
        save_dir=os.path.join(save_dir, "main/audio_geoclip/"),
        title_concat = "AG-CLIP",
    )