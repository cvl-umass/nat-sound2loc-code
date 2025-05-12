import os
import logging
import shutil
import torch
import json
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import h3 
from geoclip import LocationEncoder
import torch.nn.functional as F

from src import config

def save_model(model, save_path):
    state_dict = model.state_dict()
    torch.save(state_dict, save_path)

def save_plots(plot_dict, save_path):
    for k in plot_dict.keys():
        x = list(plot_dict[k].keys())
        y = list(plot_dict[k].values())
        if len(x) == 0: continue
        plt.plot(x, y, label=k)
    plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()

def dict_to_str(d):
    return " | ".join(["{}: {:.2f}".format(k, 100*v) for k, v in d.items()])

def metric_str2acc(m_str):
    m = m_str.replace(" ", "").split("\n")
    m = [l for l in m if l!= ""]
    m = m[1].split(":")[1].replace("%", "").replace("km", "")
    return float(m)

def write_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f)

def read_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def resolution2classes(resolution):
    res_cache = ".resolution2classes.json"
    res2cls = {}
    if os.path.exists(res_cache):
        try:
            res2cls = read_json(res_cache)      
        except KeyboardInterrupt:
            exit()
        except:
            pass
        
    
    if str(resolution) in res2cls:
        cur_classes = res2cls[str(resolution)]
        return cur_classes

    all_cells = [
        h3.latlng_to_cell(lat, lon, resolution)
        for lat in range(-90, 91) 
        for lon in range(-180, 181)
    ]
    cur_classes = sorted(list(set(all_cells)))
    res2cls[resolution] = cur_classes

    write_json(res2cls, res_cache)
    
    return cur_classes

def setup_logging(args, mode="train", exp_name=None):
    if exp_name is None:
        exp_name = "{}/{}_b{}_lr{}_wd{}/{}".format(args.model, args.optim, args.batch_size, args.lr, args.wd, mode) 
        
        if args.sound_aug:
            exp_name = "sound_aug/" + exp_name
        if args.pretrained:
            exp_name = "pretrained/" + exp_name
        if args.bce_weight > 0:
            exp_name =  "bce_{}".format(args.bce_weight) + "/" + exp_name
        if args.sinr_feat:
            exp_name =  "sinr_feat/" + exp_name
        exp_name = args.loss + "/" + exp_name
        
            

        if args.task_type == "classification": 
            exp_name = "res" + str(args.geo_resolution) + "/" + exp_name
        if args.clip_scale:
            exp_name = "clip_scale" + "/" + exp_name

        gallery_name = "uniform"
        if args.geo_gallery != "": gallery_name = args.geo_gallery
        if args.custom_gallery != "": gallery_name = args.custom_gallery
        exp_name = "gallery_" + gallery_name + "/" + exp_name

        if args.task_type == "generalclip":
            exp_name =  "_".join(args.loc_enc) + "/" + exp_name
        if args.task_type == "geo_multitask":
            exp_name =  "_".join(args.tasks) + "/" + exp_name
        exp_name =  args.task_type + "/" + exp_name
        
        if args.mixup:
            exp_name = "mixup/" + exp_name
        exp_name =  "bot{}/".format(args.bottleneck_dim) + exp_name
        
        exp_name = exp_name if args.exp_name == "" else args.exp_name + "/" + exp_name

    # get unique number of experiment
    log_root = os.path.join(config.LOG_DIR, exp_name)
    if os.path.exists(log_root):
        avail_nums = os.listdir(log_root)
        avail_nums = [-1] + [int(d) for d in avail_nums if d.isdigit()]
        log_num = max(avail_nums) + 1
    else:
        log_num = 0
    log_num = str(log_num)
    print("Logging in exp {}, number {}".format(exp_name, log_num))

    # get log directories and setup logger
    weight_dir = os.path.join(config.LOG_DIR, exp_name, log_num, "checkpoints")
    plot_dir = os.path.join(config.LOG_DIR, exp_name, log_num, "plots")
    pred_dir = os.path.join(config.LOG_DIR, exp_name, log_num, "preds")
    os.makedirs(weight_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    log_path = os.path.join(config.LOG_DIR, exp_name, log_num, "log.txt")
    logging.basicConfig(filename=log_path,
                    filemode='a',
                    format='%(asctime)s | %(message)s',
                    datefmt='%m-%d %H:%M:%S',
                    level=logging.INFO, force=True)
    logging.info("starting new experiment")

    # Copying the source python files
    src_dir = os.path.join(config.LOG_DIR, exp_name, log_num, "src")
    os.makedirs(src_dir, exist_ok=True)
    cwd = os.getcwd()
    def copy_file(rel_path):
        src = os.path.join(cwd, rel_path)
        dst = os.path.join(src_dir, rel_path)
        shutil.copy(src, dst)
    for fname in os.listdir(cwd):
        if os.path.isdir(os.path.join(cwd, fname)):
            os.makedirs(os.path.join(src_dir, fname), exist_ok=True)
            for fname2 in os.listdir(os.path.join(cwd, fname)):
                if fname2.endswith(".py"):
                    copy_file(os.path.join(fname, fname2))
        if fname.endswith(".py"):
            copy_file(fname)

    return weight_dir, plot_dir, pred_dir

    

def list2str(lst):
    if type(lst) == torch.Tensor:
        lst = lst.tolist()
    return "_".join(["{:.2f}".format(l) for l in lst])

def list_average(lst):
    return sum(lst) / len(lst) if len(lst) else 0

def add_lists(lst):
    new_list = []
    for l in lst:
        new_list.extend(l)
    return new_list

class GeoLocalizationMetrics():
    def __init__(self, task_type, geo_resolution, geo_gallery="", loc_enc=[], save_dir=None, clip_only=False, gallery_path_arg=None):
        self.count = 0
        self.task_type = task_type      # ["lat_lon", "cartesian", "classification"]
        self.geo_resolution = geo_resolution
        self.geo_classes = resolution2classes(self.geo_resolution)
        self.img2scale_distances = {}
        self.pred_locs = {}
        self.avg_pred_embed = {}
        self.cos_sim = torch.nn.CosineSimilarity(dim=-1)
        self.thresholds = config.THRESHOLDS
        self.save_dir = save_dir
        if self.save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        self.scales = ["clip", "average"]
        if self.task_type in ["classification"]:
            self.scales.append("maximum")
        if clip_only:
            self.scales = ["clip"]
        self.scales.append("clip_norm")


        self.classification_acc = {scale:[] for scale in self.scales}

        
        self.pred2latlon = None

        self.gallery_search = None
        if "search" in geo_gallery:
            self.gallery_search = 1
            geo_gallery = geo_gallery.replace("_search", "")

        if task_type == "generalclip":
            self.locations = {}
            self.loc_embeds = {}
            for enc in loc_enc:
                gallery_path = os.path.join(config.GALLERY_DIR, "{}/gallery_{}.json".format(enc, geo_gallery))
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
                self.locations[enc] = torch.Tensor(lat_lon)
                self.loc_embeds[enc] = torch.Tensor(list(loc_embeds.values()))
                self.loc_embeds[enc] = F.normalize(self.loc_embeds[enc], dim=1)
            if len(loc_enc) > 1:
                for i in range(len(loc_enc) - 1):
                    diff = torch.abs(self.locations[loc_enc[i+1]] - self.locations[loc_enc[i]])
                    assert torch.sum(diff) == 0
            self.locations = self.locations[loc_enc[0]]

            # print("Using {} Gallery".format(geo_gallery))
        elif "clip" in task_type or "multitask" in task_type:
            if geo_gallery in ["land", "train", "val"]:
                gallery_path = os.path.join(config.GALLERY_DIR, "gallery_{}.json".format(geo_gallery))
                if gallery_path_arg is not None:
                    gallery_path = gallery_path_arg
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
                # print("Using {} Gallery".format(geo_gallery))
            elif geo_gallery == "h3":
                classes = resolution2classes(self.geo_resolution)
                lat_lon = [h3.cell_to_latlng(c) for c in classes]
                self.locations = torch.Tensor(lat_lon)

                location_encoder = LocationEncoder()
                location_encoder.eval()
                self.loc_embeds = location_encoder(self.locations)
                self.loc_embeds = F.normalize(self.loc_embeds, dim=1)
            else:

                res = self.geo_resolution
                # res = 0.1
                lat_lon = [
                    (float(lat)/float(res), float(lon)/float(res))
                    for lat in range(int(-90*res), int(91*res))
                    for lon in range(int(-180*res), int(181*res))
                ]
                self.locations = torch.Tensor(lat_lon)

                location_encoder = LocationEncoder()
                location_encoder.eval()
                self.loc_embeds = location_encoder(self.locations)
                self.loc_embeds = F.normalize(self.loc_embeds, dim=1)

        

    def get_lat_lon(self, pred):
        if self.task_type == "lat_lon":
            # pred: [..., 2]
            lat_lon = pred
        elif self.task_type == "cartesian":
            # pred: [..., 3]
            lon = torch.atan2(pred[..., 1], pred[..., 0])
            r = torch.sqrt(torch.sum(pred**2, -1))
            lat = torch.acos(pred[..., 2]/(r+1e-10)) 
            lat_lon = torch.stack([
                lat * (180/np.pi) / 90,
                lon * (180/np.pi) / 180,
            ], -1)
        elif self.task_type == "classification" or self.task_type == "hierarchical":
            # pred: [..., C]
            pred_argmax = pred.argmax(-1)
            pred_classes = [self.geo_classes[p] for p in pred_argmax]
            lat_lon = torch.Tensor([
                h3.cell_to_latlng(pred_c)
                for pred_c in pred_classes
            ])
            lat_lon = lat_lon / torch.Tensor([[90, 180]])
        elif self.task_type == "generalclip":
            logits = 0
            start_dim = 0
            for enc in self.loc_embeds:
                cur_loc_embeds = self.loc_embeds[enc]
                cur_pred = pred[:, start_dim:start_dim+cur_loc_embeds.shape[1]]
                cur_pred = F.normalize(cur_pred, dim=1)
                logits += cur_pred @ cur_loc_embeds.T
            lat_lon = self.locations[logits.argmax(-1), :]
            lat_lon = lat_lon / torch.Tensor([[90, 180]])


        elif self.task_type in ["geoclip", "geo_multitask", "audio_geoclip"]:
            # pred: [N, 512]
            if self.task_type in ["geo_multitask"]:
                pred = pred[..., :512]
            pred = F.normalize(pred, dim=1)
            if self.pred2latlon is not None:
                lat_lon = self.pred2latlon(pred)
            else:
                logits =  pred @ self.loc_embeds.T # [N, locs]

                if self.gallery_search is not None:
                    lat_lon = self.gallery_search(pred, logits)
                else:
                    lat_lon = self.locations[logits.argmax(-1), :]
            
            # print(lat_lon, logits.min(), logits.max(), logits.mean(), logits.std())
            lat_lon = lat_lon / torch.Tensor([[90, 180]])

        return lat_lon

    def update_classification(self, pred_logits, labels, scale="clip"):
        label_classes = []
        for i in range(labels.shape[0]):
            lat, lon = labels[i]
            lat, lon = 90*lat, 180*lon
            cell = h3.latlng_to_cell(lat, lon, self.geo_resolution)
            label_classes.append(self.geo_classes.index(cell))
        label_classes = torch.tensor(label_classes)
        preds = pred_logits.argmax(-1)
        corrects = preds == label_classes
        # print(preds, label_classes, corrects)
        self.classification_acc[scale].extend(corrects.tolist())

    def update(self, pred_logits, labels, image_names):
        self.count += 1
        # assume all clips of a sound are in a single batch. there may be more than one sound
        
        img2pred = {}
        img2gt = {}
        if len(image_names) == 1 and (type(image_names[0]) == list or type(image_names[0]) == tuple):
            image_names = image_names[0]
        for i, img_name in enumerate(image_names):

            if img_name not in img2pred:
                img2pred[img_name] = []
            img2pred[img_name].append(pred_logits[i, :])

            if img_name not in img2gt:
                img2gt[img_name] = []
            img2gt[img_name].append(labels[i, :])
        # print(len(list(img2pred.values())))
        for img in img2pred:
            if img not in self.img2scale_distances:
                self.img2scale_distances[img] = {sc:[] for sc in self.scales}
            if img not in self.pred_locs:
                self.pred_locs[img] = {sc:[] for sc in self.scales}
            

            
            if self.pred2latlon is None and self.gallery_search is None:
                scale = self.scales[0] # clip
                lat_lon = self.get_lat_lon(torch.stack(img2pred[img], 0))
                self.pred_locs[img][scale].extend(lat_lon.tolist())
                distances = self.get_distance(
                    lat_lon, 
                    torch.stack(img2gt[img], 0)
                )
                # set nan to 0
                distances[torch.isnan(distances)] = 0
                self.img2scale_distances[img][scale].extend(distances.tolist())
                if self.task_type == "classification":
                    self.update_classification(torch.stack(img2pred[img], 0), torch.stack(img2gt[img], 0), scale)


            if len(self.scales) == 1:
                continue
            scale = self.scales[1] # average
            avg_pred = torch.mean(torch.stack(img2pred[img], 0), dim=0, keepdims=True)
            lat_lon = self.get_lat_lon(avg_pred)
            self.avg_pred_embed[img] = avg_pred.detach().cpu().numpy().tolist()
            self.pred_locs[img][scale].append(lat_lon.tolist())
            distance = self.get_distance(
                lat_lon, 
                torch.Tensor(img2gt[img][0].unsqueeze(0))
            )
            # set nan to 0
            distance[torch.isnan(distance)] = 0
            self.img2scale_distances[img][scale].extend([float(distance)])
            if self.task_type == "classification":
                self.update_classification(avg_pred, torch.Tensor(img2gt[img][0].unsqueeze(0)), scale)


            if self.task_type == "classification":

                scale = self.scales[2] # maximum
                max_pred, _ = torch.max(torch.stack(img2pred[img], 0), dim=0, keepdims=True)
                lat_lon = self.get_lat_lon(max_pred)
                distance = self.get_distance(
                    lat_lon, 
                    torch.Tensor(img2gt[img][0].unsqueeze(0))
                )
                # set nan to 0
                distance[torch.isnan(distance)] = 0
                self.img2scale_distances[img][scale].extend([float(distance)])
                self.update_classification(max_pred, torch.Tensor(img2gt[img][0].unsqueeze(0)), scale)

            
    def get_distance(self, pred, gt):
        # pred and gt need to be normalized
        lat1, lon1 = pred[..., 0] * 90, pred[..., 1] * 180
        lat2, lon2 = gt[..., 0] * 90, gt[..., 1] * 180
        r = 6371
        phi1, phi2 = torch.deg2rad(lat1), torch.deg2rad(lat2)
        delta_phi, delta_lambda = torch.deg2rad(lat2-lat1), torch.deg2rad(lon2-lon1)
        a = torch.sin(delta_phi/2)**2 + torch.cos(phi1) * torch.cos(phi2) * torch.sin(delta_lambda/2)**2
        return 2 * r * torch.asin(torch.sqrt(a))
     

    def get_values(self):
        metric_dict = {}
        if self.task_type == "classification":
            for scale in self.classification_acc:
                num = len(self.classification_acc[scale])
                class_acc = sum(self.classification_acc[scale]) / num if num != 0 else 0
                metric_dict[scale] = {"class_acc": class_acc}

        for scale in self.scales:
            if scale not in metric_dict:
                metric_dict[scale] = {}
            if scale == "clip_norm":
                all_distances = [torch.mean(torch.Tensor(self.img2scale_distances[img]["clip"])) for img in self.img2scale_distances]
                all_distances = torch.stack(all_distances)
                metric_dict[scale]["distance"] = all_distances.mean()
                metric_dict[scale]["median-distance"] = torch.median(all_distances)
                for level, thresh in self.thresholds.items():
                    acc_list = [
                        torch.mean(1.0 * (torch.Tensor(self.img2scale_distances[img]["clip"]) <= thresh))
                        for img in self.img2scale_distances
                    ]
                    metric_dict[scale][level] = sum(acc_list) / len(acc_list)
            else:
                all_distances = [torch.Tensor(self.img2scale_distances[img][scale]) for img in self.img2scale_distances]
                all_distances = torch.cat(all_distances, 0)
                metric_dict[scale]["distance"] = all_distances.mean()
                metric_dict[scale]["median-distance"] = torch.median(all_distances)
                for level, thresh in self.thresholds.items():
                    metric_dict[scale][level] = torch.sum(all_distances <= thresh) / all_distances.shape[0]

        for scale in metric_dict:
            for level in metric_dict[scale]:
                metric_dict[scale][level] = float(metric_dict[scale][level])
        return metric_dict

    def get_metric_str(self):
        if self.save_dir is not None:
            write_json(self.pred_locs, os.path.join(self.save_dir, "pred_locs.json"))
            write_json(self.img2scale_distances, os.path.join(self.save_dir, "img2scale_distances.json"))
            write_json(self.avg_pred_embed, os.path.join(self.save_dir, "avg_pred_embed.json"))
            
        metric_dict = self.get_values()
        s = "\n"
        for scale in metric_dict:
            s += "{}-level\n".format(scale)
            for k in metric_dict[scale]:
                if "distance" in k:
                    s += "|  {}: {:.2f}km\n".format(k, metric_dict[scale][k])
                else:
                    s += "|  {}: {:.2f}%\n".format(k, 100 * metric_dict[scale][k])

        s += "\n\n\n"
        csv_string = "Scale,"+",".join(list(metric_dict[self.scales[0]].keys()))
        for scale in metric_dict:
            csv_string += "\n"
            csv_string += scale + ","
            csv_string += ",".join(["{:.2f}".format(v if "dist" in k else 100*v) for k, v in metric_dict[scale].items()])
        s += csv_string
        s += "\n\n\n"
        return s

    def __str__(self):
        return self.get_metric_str()



