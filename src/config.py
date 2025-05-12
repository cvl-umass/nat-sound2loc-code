import os

# To change
### INATSOUNDS DATA
SPECTROGRAM_DIR = "/scratch3/workspace/mchasmai_umass_edu-inat_sound/data/inatsounds_release/mel_np_corrected"
JSON_DIR = "/scratch3/workspace/mchasmai_umass_edu-inat_sound/data/inatsounds_release"
LOG_DIR = "/scratch3/workspace/mchasmai_umass_edu-inat_sound/output_logs/sound2loc/main_neurips/reproduce"

### XCDC DATA
XCDC_SPECTROGRAM_DIR = "/scratch3/workspace/mchasmai_umass_edu-inat_sound/data/dawn_chorus/xeno_canto_v2/mel_np"
XCDC_CSV="/scratch3/workspace/mchasmai_umass_edu-inat_sound/data/dawn_chorus/xeno_canto_v2/neurips_submission/xcdc_recordings.csv"

GALLERY_DIR = "/scratch3/workspace/mchasmai_umass_edu-inat_sound/data/galleries"

OCEAN_MASK = "/scratch3/workspace/mchasmai_umass_edu-inat_sound/data/geo_model/data/masks/ocean_mask.npy"



### Model Weights
SATCLIP_WEIGHTS = "/home/mchasmai_umass_edu/sound2loc/satclip-vit16-l40.ckpt"
SINR_WEIGHTS = "/scratch3/workspace/mchasmai_umass_edu-inat_sound/data/geo_model/experiments/train_inatsound_classes_only/model.pt"

# Probably will not need to change
THRESHOLDS = {
    "street": 1,
    "city": 25,
    "region": 200,
    "country": 750,
    "continent": 2500,
}

TRAIN_JSON = os.path.join(JSON_DIR, "train.json")
VAL_JSON = os.path.join(JSON_DIR, "val.json")
TEST_JSON = os.path.join(JSON_DIR, "test.json")
JSON_DICT = {
    "train": TRAIN_JSON,
    "val": VAL_JSON,
    "test": TEST_JSON,
}