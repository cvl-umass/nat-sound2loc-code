import torchvision
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import json
import h3



from src import utils, config
from geoclip import LocationEncoder
try:
    from satclip.load import get_satclip
except ModuleNotFoundError:
    pass

hidden_dim_dict = {
    "resnet18": 512,
    "resnet50": 2048,
    "resnet101": 2048,
    "ensemble": 2048,
    "vit": 768,
    "swin": 768,
    "mobilenet": 1280,
}

def get_model(model_name, output_dim, pretrained=True, dropout=0, transformer=False, fast_tr=False, args=None):
    bottleneck_dim = 512
    if args is not None:
        bottleneck_dim = args.bottleneck_dim
        
    last_dim = hidden_dim_dict[model_name] if model_name in hidden_dim_dict else None
    if output_dim is not None and last_dim is not None:
        if bottleneck_dim is not None:
            cls_head = nn.Sequential(
                nn.Linear(last_dim, bottleneck_dim),
                nn.ReLU(),
                nn.Linear(bottleneck_dim, output_dim)
            )
        else:
            cls_head = nn.Linear(last_dim, output_dim)
    else:
        cls_head = nn.Identity()
    if model_name == "resnet18":
        model = torchvision.models.resnet18(pretrained=pretrained)
        model.fc = cls_head
    elif model_name == "resnet50":
        model = torchvision.models.resnet50(pretrained=pretrained)
        model.fc = cls_head
    elif model_name == "resnet101":
        model = torchvision.models.resnet101(pretrained=pretrained)
        model.fc = cls_head
    elif model_name == "vit":
        model = torchvision.models.vit_b_16(pretrained=pretrained)
        model.heads.head = cls_head
    elif model_name == "mobilenet":
        model = torchvision.models.mobilenet_v3_large(pretrained=pretrained)
        model.classifier[3] = cls_head
    elif model_name == "swin":
        model = torchvision.models.swin_s()
        model.head = nn.Linear(last_dim, output_dim) if output_dim is not None else nn.Identity()

    elif model_name == "hierarchical":
        run = 2
        model_list = [
            (
                "mobilenet",
                0.0, 122,
                "/scratch3/workspace/mchasmai_umass_edu-inat_sound/output_logs/sound2loc/main_neurips/inat_pre/bot128/classification/gallery_uniform/res0.1/pretrained/sound_aug/mobilenet/nesterov_b128_lr0.05_wd1e-05/train/{}/checkpoints/checkpoint_49.pt".format(run)
            ),
            (
                "mobilenet",
                1.0, 842,
                "/scratch3/workspace/mchasmai_umass_edu-inat_sound/output_logs/sound2loc/main_neurips/inat_pre/bot128/classification/gallery_uniform/res1.0/pretrained/sound_aug/mobilenet/nesterov_b128_lr0.05_wd1e-05/train/{}/checkpoints/checkpoint_49.pt".format(run)
            ),
            (
                "mobilenet",
                2.0, 5882,
                "/scratch3/workspace/mchasmai_umass_edu-inat_sound/output_logs/sound2loc/main_neurips/inat_pre/bot128/classification/gallery_uniform/res2.0/pretrained/sound_aug/mobilenet/nesterov_b128_lr0.05_wd1e-05/train/{}/checkpoints/checkpoint_49.pt".format(run)
            )
        ]
        model = HierarchihcalModel(model_list, args)

    return model

###### Taken from SINR: https://github.com/elijahcole/sinr

class CoordEncoder:

    def __init__(self, input_enc, raster=None):
        self.input_enc = input_enc
        self.raster = raster

    def encode(self, locs, normalize=True):
        # assumes lat, lon in range [-90, 90] and [-180, 180] 
        locs = locs.clone()
        locs = torch.flip(locs, [-1])
        # assumes lon, lat in range [-180, 180] and [-90, 90]
        locs = locs.clone()
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

###### SINR ends here 


class HierarchihcalModel(nn.Module):
    def __init__(self, model_list, args):
        super().__init__()
        self.models = {}
        resolutions = []
        for model_name, res, num_classes, weight_path in model_list:
            model = get_model(model_name, num_classes, pretrained=False, args=args)
            weight = torch.load(weight_path)
            model.load_state_dict(weight)
            model.eval()
            self.models[str(res).replace(".", "_")] = model
            resolutions.append(float(res))

        self.models = nn.ModuleDict(self.models)
        self.resolutions = sorted(resolutions)

        self.res2classes = {}
        for res in resolutions:
            self.res2classes[res] = utils.resolution2classes(res)
        
        self.res_class_masks = {}
        for i in range(len(resolutions) - 1):
            res1 = resolutions[i]
            res2 = resolutions[i+1]

            classes1 = self.res2classes[res1]
            classes2 = self.res2classes[res2]

            self.res_class_masks[res2] = torch.zeros(len(classes1), len(classes2))
            for c1_i, c1 in enumerate(classes1):
                for c2_i, c2 in enumerate(classes2):
                    lat, lon = h3.cell_to_latlng(c2)
                    covering_c1 = h3.latlng_to_cell(lat, lon, res1)
                    if covering_c1 == c1:
                        self.res_class_masks[res2][c1_i][c2_i] = 1
            self.res_class_masks[res2] = self.res_class_masks[res2].cuda()

    def forward(self, x):
        prev_pred = None
        for resolution in self.resolutions:
            logits = self.models[str(resolution).replace(".", "_")](x)
            if resolution in self.res_class_masks and prev_pred is not None:
                mask = self.res_class_masks[resolution][prev_pred]
                logits = logits * mask + (logits.min() - 10) * (1-mask)
            prev_pred = logits.argmax(-1)
        return logits
            
            


def geodesic_distance(loc1, loc2, normalized=True):
    lat1, lon1 = loc1[..., 0] * 90, loc1[..., 1] * 180
    lat2, lon2 = loc2[..., 0] * 90, loc2[..., 1] * 180
    if normalized:
        r = 1/np.pi  # 2* Radius of Earth / circumference
    else:
        r = 6371
    phi1, phi2 = torch.deg2rad(lat1), torch.deg2rad(lat2)
    delta_phi, delta_lambda = torch.deg2rad(lat2-lat1), torch.deg2rad(lon2-lon1)
    a = torch.sin(delta_phi/2)**2 + torch.cos(phi1) * torch.cos(phi2) * torch.sin(delta_lambda/2)**2
    return 2 * r * torch.asin(torch.sqrt(a))



def get_opt(args, model):
    if args.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim == "sgd_momentum":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
    elif args.optim == "nesterov":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9, nesterov=True)
    elif args.optim == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        raise NotImplementedError

    return optimizer
    

class GeographicLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # pred: [..., 2], target: [..., 2]
        lat1, lon1 = pred[..., 0] * 90, pred[..., 1] * 180
        lat2, lon2 = target[..., 0] * 90, target[..., 1] * 180
        r = 1/np.pi  # 2* Radius of Earth / circumference
        phi1, phi2 = torch.deg2rad(lat1), torch.deg2rad(lat2)
        delta_phi, delta_lambda = torch.deg2rad(lat2-lat1), torch.deg2rad(lon2-lon1)
        a = torch.sin(delta_phi/2)**2 + torch.cos(phi1) * torch.cos(phi2) * torch.sin(delta_lambda/2)**2
        return torch.mean(2 * r * torch.asin(torch.sqrt(a)))

class GeoClassLoss(nn.Module):
    def __init__(self, resolution):
        super().__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.resolution = resolution
        self.geo_classes = utils.resolution2classes(self.resolution)
        self.geo_class2idx = {c:i for i, c in enumerate(self.geo_classes)}

    def forward(self, pred, target):
        # pred: [..., C], target: [..., 2]
        
        target_class = [
            self.geo_class2idx[
                h3.latlng_to_cell(
                    90*target[i, 0], 180*target[i, 1], 
                    self.resolution
                )
            ]
            for i in range(target.shape[0])
        ]
        target_class = torch.Tensor(target_class).to(torch.long).to(target.device)

        return self.ce_loss(pred, target_class)

class GeoCartesianLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, pred, target):
        # pred: [..., 3], target: [..., 2]

        # convert target from lat, lon to x, y, z
        # Convention: z: North, x: lon=0
        lat, lon = target[..., 0] * (np.pi/2), target[..., 1] * (np.pi)
        target_x = torch.cos(lat) * torch.cos(lon)
        target_y = torch.cos(lat) * torch.sin(lon)
        target_z = torch.sin(lat)
        target_cartesian = torch.stack([target_x, target_y, target_z], -1)

        # pred_radius = torch.sum(pred * pred)
        # return self.mse_loss(pred, target_cartesian) + 0.1 * torch.abs(1 - pred_radius)

        return self.mse_loss(pred, target_cartesian) 

class GeoClipLoss(nn.Module):
    def __init__(self, clip_scale=False):
        super().__init__()
        self.clip_scale = clip_scale
        self.location_encoder = LocationEncoder().cuda()
        self.location_encoder.eval()
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.lat_lon_scale = torch.Tensor([[90.0, 180.0]]).cuda()

    def forward(self, pred, target, logit_scale=None, loc_mlp=None):
        # pred: [N, 512], target: [N, 2]

        location_features = self.location_encoder(target * self.lat_lon_scale) # N x 512
        location_features = location_features.detach().clone()
        if loc_mlp is not None:
            location_features = loc_mlp(location_features)

        pred = F.normalize(pred, dim=1)
        location_features = F.normalize(location_features, dim=1)

        logits_per_image = pred @ location_features.t() 
        if logit_scale is not None:
            logits_per_image = logit_scale.exp() * logits_per_image 
        elif self.clip_scale:
            logits_per_image = 3.6810 * logits_per_image
            # logits_per_image = 100 * logits_per_image
        clip_labels = torch.Tensor([i for i in range(target.shape[0])]).long().to(target.device)

        return self.ce_loss(logits_per_image, clip_labels) 

class GeneralClipLoss(nn.Module):
    def __init__(self, encoders=["geoclip", "satclip", "sinr"], clip_scale=False):
        super().__init__()
        self.clip_scale = clip_scale
        self.location_encoder = {enc: None for enc in encoders}
        if "geoclip" in encoders:
            self.location_encoder["geoclip"] = LocationEncoder().cuda().eval()
        if "satclip" in encoders:
            satclip_model = get_satclip(config.SATCLIP_WEIGHTS, device="cuda").eval()
            def satclip_encoder(loc):
                return satclip_model(torch.flip(loc, [-1]).double())
            self.location_encoder["satclip"] = satclip_encoder
        if "sinr" in encoders:
            enc = CoordEncoder("sin_cos", raster=None)
            sinr_model = ResidualFCNet(4, 5547, 256, 4).eval().cuda()
            model_path = config.SINR_WEIGHTS
            sinr_model.load_state_dict(torch.load(model_path, map_location='cpu')['state_dict'], strict=True)
            def sinr_encoder(loc):
                locs_enc = enc.encode(loc).cuda()
                return sinr_model(locs_enc, return_feats=True)
            self.location_encoder["sinr"] = sinr_encoder
            
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.lat_lon_scale = torch.Tensor([[90.0, 180.0]]).cuda()

    def forward(self, pred, target, logit_scale=None, loc_mlp=None):
        # pred: [N, 512], target: [N, 2]
        clip_labels = torch.Tensor([i for i in range(target.shape[0])]).long().to(target.device)

        loss = 0
        start_dim = 0
        for encoder in self.location_encoder:
            cur_embeds = self.location_encoder[encoder](target * self.lat_lon_scale)
            cur_embeds = cur_embeds.detach().clone()
            cur_embeds = cur_embeds.float()
            if loc_mlp is not None:
                cur_embeds = loc_mlp(cur_embeds)
            
            cur_pred = pred[:, start_dim:start_dim+cur_embeds.shape[1]]
            start_dim = start_dim + cur_embeds.shape[1]
            
            cur_embeds = F.normalize(cur_embeds, dim=1)
            cur_pred = F.normalize(cur_pred, dim=1)

            cur_logits = cur_pred @ cur_embeds.t() 
            if logit_scale is not None:
                logits_per_image = logit_scale.exp() * logits_per_image 
            elif self.clip_scale:
                cur_logits = 3.6810 * cur_logits
            loss += self.ce_loss(cur_logits, clip_labels) 


        return loss



class AGClipLoss(nn.Module):
    def __init__(self, clip_scale, sinr_feat=False, bce_weight=0.01):
        super().__init__()
        self.species_dim = 5547
        self.sinr_feat = sinr_feat
        self.bce_weight= bce_weight
        self.clip_scale = clip_scale

        self.location_encoder = LocationEncoder().cuda()
        self.location_encoder.eval()
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.lat_lon_scale = torch.Tensor([[90.0, 180.0]]).cuda()


        # SINR stuff
        self.enc = CoordEncoder("sin_cos", raster=None)
        self.sinr_model = ResidualFCNet(4, 5547, 256, 4).eval().cuda()
        model_path = config.SINR_WEIGHTS
        self.sinr_model.load_state_dict(torch.load(model_path, map_location='cpu')['state_dict'], strict=True)
        self.sinr_model.eval()


        self.bce_loss_fn = nn.BCELoss(reduction="none")
        # self.bce_loss_fn = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()
        self.mse_loss = torch.nn.MSELoss()

    def get_sinr_species(self, loc):
        locs_enc = self.enc.encode(loc).cuda()
        if self.sinr_feat:
            species_feat = self.sinr_model(locs_enc, return_feats=True)
            return species_feat
        else:
            species_logits = self.sinr_model(locs_enc, return_feats=False)
            species_one_hot = 1.0 * (species_logits > 0.1)
            return species_one_hot

    def forward(self, pred, target, loc_mlp=None):
        # pred: [N, 5547 + 512], target: [N, 2]
        if self.sinr_feat:
            pred_species_raw = pred[..., :256]
            pred_geoclip = pred[..., 256:]
            pred_species = pred_species_raw
        else:
            pred_species_raw = pred[..., :self.species_dim]
            pred_geoclip = pred[..., self.species_dim:]
            pred_species = self.sigmoid(pred_species_raw)
            


        loss = 0
        batch_size = pred.shape[0]
        sinr_species = self.get_sinr_species(target * self.lat_lon_scale)
        if self.sinr_feat:

            pred_species = F.normalize(pred_species, dim=1)
            sinr_species = F.normalize(sinr_species, dim=1)

            clip_labels = torch.Tensor([i for i in range(target.shape[0])]).long().to(target.device)
            loss1 = self.ce_loss(pred_species @ sinr_species.t(), clip_labels)

        else:
            loss1 = self.bce_loss_fn(pred_species, sinr_species)
            loss1 = loss1.sum() / batch_size

 
        location_features = self.location_encoder(target * self.lat_lon_scale) # N x 512
        location_features = location_features.detach().clone()
        if loc_mlp is not None:
            location_features = loc_mlp(location_features)

        

        pred_geoclip = F.normalize(pred_geoclip, dim=1)
        location_features = F.normalize(location_features, dim=1)

        logits_per_image = pred_geoclip @ location_features.t() 
        if self.clip_scale:
            logits_per_image = 3.6810 * logits_per_image
        
        clip_labels = torch.Tensor([i for i in range(target.shape[0])]).long().to(target.device)

        loss2 = self.ce_loss(logits_per_image, clip_labels) 

        if self.sinr_feat:
            loss += loss1 + loss2
        else:
            loss += self.bce_weight * loss1 + loss2
        return loss


    

class GeoMultiTaskLoss(nn.Module):
    def __init__(self, clip_scale=False, resolution=0.1, tasks=[]):
        super().__init__()
        self.loss_fns = []
        if "geoclip" in tasks:
            self.loss_fns.append(
                ("geoclip", GeoClipLoss(clip_scale=clip_scale))
            )
        if "classification" in tasks:
            cls_loss = GeoClassLoss(resolution)
            self.loss_fns.append(("classification", cls_loss))
        if "lat_lon" in tasks:
            self.loss_fns.append(("lat_lon", GeographicLoss()))
        if "cartesian" in tasks:
            self.loss_fns.append(("cartesian", GeoCartesianLoss()))
        self.task2dim = {
            "lat_lon": 2,
            "cartesian": 3,
            "classification": len(cls_loss.geo_classes) if "classification" in tasks else None,
            "geoclip": 512,
        }
        
    def forward(self, pred, target):
        loss = 0
        start_dim = 0
        
        for task_name, loss_fn in self.loss_fns:
            cur_dim = self.task2dim[task_name]
            loss += loss_fn(pred[..., start_dim:start_dim+cur_dim], target)
            start_dim = start_dim + cur_dim

        return loss

class MixupLoss(nn.Module):
    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, pred, target):
        target_dim = target.shape[-1]
        target_single_dim = (target_dim - 1) // 2
        loss1 = self.loss_fn(pred, target[..., :target_single_dim])
        loss2 = self.loss_fn(pred, target[..., target_single_dim:2*target_single_dim])
        lam = target[..., -1].mean()
        loss = loss1 * lam + loss2 * (1.0 - lam)
        return loss

def get_loss(args):
    if args.task_type == "lat_lon":
        if args.loss == "mse":
            loss_fn = torch.nn.MSELoss()
        else:
            loss_fn = GeographicLoss()
    elif args.task_type == "cartesian":
        loss_fn = GeoCartesianLoss()
    elif args.task_type == "classification":
        loss_fn = GeoClassLoss(args.geo_resolution)
    elif args.task_type == "geoclip":
        loss_fn = GeoClipLoss(clip_scale=args.clip_scale)
    elif args.task_type == "generalclip":
        loss_fn = GeneralClipLoss(encoders=args.loc_enc, clip_scale=args.clip_scale)
    elif args.task_type=="audio_geoclip":
        loss_fn = AGClipLoss(clip_scale=args.clip_scale, sinr_feat=args.sinr_feat, bce_weight=args.bce_weight)
    elif args.task_type == "geo_multitask":
        loss_fn = GeoMultiTaskLoss(clip_scale=args.clip_scale, resolution=args.geo_resolution, tasks=args.tasks)
    else:
        raise NotImplementedError

    return loss_fn