import matplotlib.pyplot as plt
import numpy as np
import utils
import os
# import scienceplots
# plt.style.use(['science','ieee', "no-latex", "high-vis"])
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']


def plot_cdf(predictions, title, clip=False, add_num=False, linestyle="-", ax=None):
    metric = "clip" if clip else "average"
    if metric == "average" and metric not in predictions[list(predictions.keys())[0]]:
        metric = "clip"
    distances = np.array([predictions[k][metric][0] for k in predictions])

    distances = np.sort(distances)
    # make cdf of distances
    cdf = np.linspace(0, 1, len(distances))
    if add_num:
        title = "All (n={})".format(len(distances))
    if ax is None:
        plt.plot(distances, 100*cdf, label=title, linestyle=linestyle)
    else:
        ax.plot(distances, 100*cdf, label=title, linestyle=linestyle)


plt.figure(figsize=(5.5, 4))
thresholds = {
    # "street": 1,
    "city": 25,
    "region": 200,
    "country": 750,
    "continent": 2500,
}



label_preds = [
    ("Species Oracle (50% Corrupted)", "/home/mchasmai_umass_edu/sound2loc/oracles/test/run1/geoclip_species_0.1_0.1_corruption_0.5/img2scale_distances.json"),
    ("Naive (Random Train Loc)", "/home/mchasmai_umass_edu/sound2loc/trivial_baselines/test/random_pick/img2scale_distances.json"),
    ("Species Ranges (Predicted All)", "/home/mchasmai_umass_edu/sound2loc/dawn_chorus/preds_sinr_intersect/avg_pred/img2scale_distances.json"),
    ("Classification (Hierarchical)", "/scratch3/workspace/mchasmai_umass_edu-inat_sound/output_logs/sound2loc/main_neurips/inat_pre/bot128/classification/gallery_uniform/res2.0/pretrained/sound_aug/hierarchical/nesterov_b128_lr0.05_wd1e-05/eval/0/preds/test/img2scale_distances.json"),
    ("Retrieval (AG-CLIP) (ours)", "/scratch3/workspace/mchasmai_umass_edu-inat_sound/output_logs/sound2loc/main_neurips/inat_pre/bot128/audio_geoclip/gallery_train/clip_scale/bce_0.01/pretrained/sound_aug/mobilenet/nesterov_b128_lr0.05_wd1e-05/train/2/preds/test/epoch_49/img2scale_distances.json"),    
]
plt.figure(figsize=(3.4, 2))
# plt.figure(figsize=(2.56, 1.5))

for label, pred_path in label_preds:
    preds = utils.read_json(pred_path)
    # linestyle = "-" if "Ours" in label or "AG-CLIP" in label else "--"
    linestyle = "--" if "Chk" in label or "Random" in label else "-"
    plot_cdf(preds, label, clip="Rnd" in label or "Random" in label or "Chk" in label or "Checklist" in label, linestyle=linestyle)



for name, th in thresholds.items():
    plt.axvline(x = th, linestyle="dotted")
    text = plt.text(th*1.05, 0.0, name,rotation=90, va="bottom")
    text.set_bbox(dict(facecolor='white', edgecolor='none', alpha=0.9))
for y in [25, 50, 75]:
    plt.axhline(y = y, color="0.9", zorder=-10)

plt.xlabel("Distance (km)")
plt.ylabel("Correctly Geolocated (%)")
# plt.xlim(0.1, 5e4)
plt.xlim(10, 3e4)
plt.yticks([25, 50, 75])
plt.xscale("log")
# plt.legend(bbox_to_anchor=(1.01, 0.5), loc='center left')
legend = plt.legend(bbox_to_anchor=(1, 0.5), loc='center left', facecolor='white', framealpha=0.9, labelspacing=1.2)

# Get the maximum width of legend texts
max_width = max([t.get_window_extent().width for t in legend.get_texts()])

# Adjust text alignment and position
# for t in legend.get_texts():
#     t.set_ha('center')
#     t.set_position((max_width - t.get_window_extent().width, 0))


plt.title("CDF of Geolocation Errors", fontsize=15)
plt.savefig("all_cdfs/main/cdf_models.pdf", bbox_inches='tight', dpi=300, pad_inches=0)
plt.savefig("all_cdfs/main/cdf_models.png", bbox_inches='tight', dpi=300, pad_inches=0)
plt.close()





