import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import h3
import tqdm
import argparse

from src import models, utils, config


parser = argparse.ArgumentParser()
parser.add_argument('--eval_set', type=str, default="test")
parser.add_argument('--run', type=int, default=1)
parser.add_argument('--save_dir', type=str, default="./species_oracles")
args = parser.parse_args()

EVAL_SET = args.eval_set

MODEL_SAVE_DIR = os.path.join(args.save_dir, "run{}".format(args.run))
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

PLOT_DIR = os.path.join(args.save_dir, "run{}".format(args.run))
os.makedirs(PLOT_DIR, exist_ok=True)

class CoordEncoder:

    def __init__(self, input_enc, raster=None):
        self.input_enc = input_enc
        self.raster = raster

    def encode(self, locs, normalize=True):
        # assumes lat, lon in range [-90, 90] and [-180, 180] 
        locs = locs.clone()
        locs = torch.flip(locs, [-1])
        # assumes lon, lat in range [-180, 180] and [-90, 90]
        if normalize:
            locs = normalize_coords(locs)
        if self.input_enc == 'sin_cos': # sinusoidal encoding
            loc_feats = encode_loc(locs)
        else:
            raise NotImplementedError('Unknown input encoding.')
        return loc_feats

def normalize_coords(locs):
    # locs is in lon {-180, 180}, lat {90, -90}
    # output is in the range [-1, 1]

    locs[:,0] /= 180.0
    locs[:,1] /= 90.0

    return locs

def encode_loc(loc_ip, concat_dim=1):
    # assumes inputs location are in range -1 to 1
    # location is lon, lat
    feats = torch.cat((torch.sin(np.pi*loc_ip), torch.cos(np.pi*loc_ip)), concat_dim)
    return feats

def get_model(params):

    if params['model'] == 'ResidualFCNet':
        return ResidualFCNet(params['input_dim'], params['num_classes'], params['num_filts'], params['depth'])
    else:
        raise NotImplementedError('Invalid model specified.')

class ResLayer(nn.Module):
    def __init__(self, linear_size):
        super(ResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y
        return out

class ResidualFCNet(nn.Module):

    def __init__(self, num_inputs, num_classes, num_filts, depth=4):
        super(ResidualFCNet, self).__init__()
        self.inc_bias = False
        self.class_emb = nn.Linear(num_filts, num_classes, bias=self.inc_bias)
        layers = []
        layers.append(nn.Linear(num_inputs, num_filts))
        layers.append(nn.ReLU(inplace=True))
        for i in range(depth):
            layers.append(ResLayer(num_filts))
        self.feats = torch.nn.Sequential(*layers)

    def forward(self, x, class_of_interest=None, return_feats=False):
        loc_emb = self.feats(x)
        if return_feats:
            return loc_emb
        if class_of_interest is None:
            class_pred = self.class_emb(loc_emb)
        else:
            class_pred = self.eval_single_class(loc_emb, class_of_interest)
        return torch.sigmoid(class_pred)

    def eval_single_class(self, x, class_of_interest):
        if self.inc_bias:
            return x @ self.class_emb.weight[class_of_interest, :] + self.class_emb.bias[class_of_interest]
        else:
            return x @ self.class_emb.weight[class_of_interest, :]


class LocalizeModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            # nn.Linear(in_dim, 512),
            # nn.ReLU(),
            # nn.Linear(512, out_dim),
        )

    def forward(self, x):
        return self.layers(x)

def get_checklist(cur_feat, corruption, threshold, mode):
    # for i in range(cur_feat.shape[0]):
    #     num_positives.append(cur_feat[i].sum().item())
    if mode == "species" and threshold is not None:
        if corruption is not None and type(corruption) == str:
            keep_top = corruption.replace("top", "")
            keep_top = int(keep_top)
            # keep top corruption species, mask rest to 0
            for i in range(cur_feat.shape[0]):
                top_indices = torch.topk(cur_feat[i], keep_top, largest=True).indices
                cur_feat[i, :] = 0
                cur_feat[i, top_indices] = 1
        
        cur_feat = 1.0 * (cur_feat >= threshold)
        if corruption is not None and type(corruption) != str:
            # cur_feat is N x 5574
            for i in range(cur_feat.shape[0]):
                positive_indices = torch.where(cur_feat[i] == 1)[0]     # 5574
                negative_indices = torch.where(cur_feat[i] == 0)[0]


                # if corruption < 0:
                #     corruption = -corruption
                #     editing_indices = negative_indices
                #     set_to = 1
                # else:
                #     editing_indices = positive_indices
                #     set_to = 0
                editing_indices = positive_indices
                set_to = 0


                if corruption < 1:
                    num_corruptions = int(corruption * editing_indices.shape[0])
                else:
                    num_corruptions = int(editing_indices.shape[0] - corruption)
                    num_corruptions = max(0, num_corruptions)
                if num_corruptions > 0:
                    corrupted_idx = np.random.choice(editing_indices.cpu().numpy(), num_corruptions, replace=False)
                    cur_feat[i, corrupted_idx] = set_to
                    
        # print(cur_feat.min(), cur_feat.max())
    return cur_feat

def main(mode, resolution, threshold=None, task_type="classification", uniform_train=False, eval_only=False, corruption=None):

    model_save_dir = MODEL_SAVE_DIR
    if uniform_train:
        model_save_dir = os.path.join(model_save_dir, "uniform_train")

    plot_save_dir = PLOT_DIR
    if uniform_train:
        plot_save_dir = os.path.join(plot_save_dir, "uniform_train")

    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(plot_save_dir, exist_ok=True)

    model_path = config.SINR_WEIGHTS

    eval_params = {}
    if 'device' not in eval_params:
        eval_params['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eval_params['model_path'] = model_path

    # load model
    train_params = torch.load(eval_params['model_path'], map_location='cpu')
    sinr_model = get_model(train_params['params'])
    sinr_model.load_state_dict(train_params['state_dict'], strict=True)
    sinr_model = sinr_model.to(eval_params['device'])
    sinr_model.eval()


    raster = None
    enc = CoordEncoder(train_params['params']['input_enc'], raster=raster)





    with open(config.TRAIN_JSON, "r") as f:
        train_data = json.load(f)
    eval_json = config.VAL_JSON if EVAL_SET == "val" else config.TEST_JSON
    with open(eval_json, "r") as f:
        eval_data = json.load(f)
    all_locs = [
        [a["latitude"], a["longitude"]]
        for a in train_data["audio"]
    ]
    all_locs = torch.Tensor(all_locs)
    locs_enc = enc.encode(all_locs).to(eval_params['device'])
    lat_lon_scale = torch.Tensor([[90.0, 180.0]])


    geo_classes = utils.resolution2classes(resolution)
    if task_type == "classification":
        loss_fn = models.GeoClassLoss(resolution)
    else:
        loss_fn = models.GeoClipLoss(clip_scale=True)

    all_locs = all_locs / lat_lon_scale
    all_locs = all_locs.cuda()
    out_dim = len(geo_classes) if task_type == "classification" else 512
    if mode != "species":
        loc_model = LocalizeModel(256, out_dim).cuda()
        # loc_model = ResidualFCNet(256, len(geo_classes), 256, 4).cuda()
    else:
        loc_model = LocalizeModel(5547, out_dim).cuda()
        # loc_model = LocalizeModel(47375, out_dim).cuda()
        # loc_model = ResidualFCNet(47375, len(geo_classes), 256, 4).cuda()
    
    exp_name = "{}_{}_{}{}".format(task_type, mode, resolution, "" if threshold is None else "_{}".format(threshold))
    # exp_name = "2layer_{}_{}{}".format(task_type, mode, "" if threshold is None else "_{}".format(threshold))

    model_weight_path = os.path.join(model_save_dir, "{}.pt".format(exp_name))
    if not eval_only and corruption is not None:
        model_weight_path = os.path.join(model_save_dir, "{}_corruption_{}.pt".format(exp_name, corruption))

    if not eval_only:
        loc_model.train()
        if task_type == "classification":
            optimizer = torch.optim.Adam(loc_model.parameters(), lr=1e-2)
        else:
            optimizer = torch.optim.Adam(loc_model.parameters(), lr=1e-2)
        batch_size = 2048
        all_losses = []
        num_species = []
        for epoch in tqdm.tqdm(range(100), leave=False):
        # for epoch in range(10):
            avg_loss = []
            for batch_iter in tqdm.tqdm(range(all_locs.shape[0]//batch_size), leave=False):
            # for batch_iter in range(all_locs.shape[0]//batch_size):
                optimizer.zero_grad()

                batch_idx = np.random.choice(all_locs.shape[0], batch_size)
                if uniform_train:
                    batch_lat_lon = 2*torch.rand(batch_size, 2) - 1
                    batch = enc.encode(lat_lon_scale * batch_lat_lon).cuda()
                    batch_lat_lon = batch_lat_lon.cuda()
                else:
                    batch = locs_enc[batch_idx, :]
                    batch_lat_lon = all_locs[batch_idx, :] 

                with torch.no_grad():
                    batch_species = sinr_model(batch, return_feats=mode != "species")

                # if mode == "species" and threshold is not None:
                #     # print(batch_species.min(), batch_species.max(), batch_species.mean())
                #     batch_species = 1.0 * (batch_species >= threshold)
                #     # print(batch_species.min(), batch_species.max(), batch_species.mean())
                #     num_species.extend(batch_species.sum(-1).tolist())
                batch_species = get_checklist(batch_species, corruption, threshold, mode)

                
                geo_pred = loc_model(batch_species)
                loss = loss_fn(geo_pred, batch_lat_lon)
                
                loss.backward()
                optimizer.step()

                avg_loss.append(loss.item())
                
            # num_species = sorted(num_species)
            # plt.plot(num_species)
            # plt.xlabel("Train recording")
            # plt.ylabel("Number of species given by SINR, threshold={}".format(threshold))
            # plt.savefig("num_species_{}.png".format(threshold))
            # plt.close()
            # print("Average num_species", sum(num_species)/len(num_species))
            # return 
            all_losses.append(sum(avg_loss) / len(avg_loss))
            # print(all_losses[-1])
            plt.plot(all_losses)
            plt.savefig(os.path.join(plot_save_dir, "losses_{}.png".format(exp_name)))

            # plt.savefig(os.path.join(PLOT_DIR, "loss.png"))
            plt.close()
                
        torch.save(loc_model.state_dict(), model_weight_path)
    else:
        loc_model.load_state_dict(torch.load(model_weight_path))


    img_names = [a["file_name"] for a in eval_data["audio"]]
    all_eval_locs = [
        [a["latitude"], a["longitude"]]
        for a in eval_data["audio"]
    ]
    all_eval_locs = torch.Tensor(all_eval_locs) 
    locs_enc = enc.encode(all_eval_locs).to(eval_params['device'])
    all_eval_locs = all_eval_locs / lat_lon_scale
    metrics = utils.GeoLocalizationMetrics(
        task_type, 
        geo_resolution=resolution, 
        geo_gallery="train",
        clip_only=True
    )
    

    # return 
    loc_model.eval()
    batch_size = 128
    # num_positives = []
    with torch.no_grad():
        num_batches = (locs_enc.shape[0] + batch_size - 1) // batch_size
        return_feats = mode != "species"
        # return_feats = True
        val_geo = []
        for batch_i in tqdm.tqdm(range(num_batches)):
            cur_loc_enc = locs_enc[batch_i*batch_size: min(locs_enc.shape[0], (batch_i+1)*batch_size)]
            cur_feat = sinr_model(cur_loc_enc, return_feats=return_feats)
            cur_feat = get_checklist(cur_feat, corruption, threshold, mode)
            cur_geo = loc_model(cur_feat)
            # print(cur_geo.shape)
            # print(cur_geo.min(), cur_geo.max())
            val_geo.append(cur_geo)
        val_geo = torch.cat(val_geo, 0)
    
    metrics.update(
        val_geo.cpu(), all_eval_locs.cpu(), img_names
    )
    if uniform_train:
        print("Uniform train!")

    print("data: {}, mode: {}, threshold: {}, task_type: {}, corruption: {}".format(EVAL_SET, mode, threshold, task_type, corruption))
    print(metrics)


if __name__ == "__main__":


    main(mode = "species", resolution=0.1, task_type="geoclip", threshold=0.1, corruption=0.5)
    main(mode = "species", resolution=0.1, task_type="geoclip", threshold=0.1, corruption=0.9)
    main(mode = "species", resolution=0.1, task_type="geoclip", threshold=0.1, corruption=10)
    main(mode = "species", resolution=0.1, task_type="geoclip", threshold=0.1, corruption=5)
