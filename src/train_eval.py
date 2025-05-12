import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import json
import h3

from src import utils

def run_loop(args, dataloader, model, loss_fn, mode="train", optimizer=None, use_cuda=True, save_dir=None, epoch=50):
    num_classes = dataloader.num_classes
    sigmoid = nn.Sigmoid()
    softmax = nn.Softmax(dim=-1)
    if mode == "train":
        model.train()
    else:
        model.eval()
        metrics = utils.GeoLocalizationMetrics(
            args.task_type, 
            geo_resolution=args.geo_resolution, 
            geo_gallery=args.geo_gallery,
            loc_enc=args.loc_enc,
            save_dir=save_dir,
            gallery_path_arg=None if args.custom_gallery=="" else args.custom_gallery,
        )


    avg_loss = []

    if args.iter_fraction < 1.0 and mode == "train":
        max_iters = len(dataloader) * args.iter_fraction
    else:
        max_iters = len(dataloader)
    batch_pbar = tqdm.tqdm(total=max_iters, desc=mode + " batches", leave=False)
    for batch_id, batch in enumerate(dataloader):

        if batch_id >= max_iters:
            break

        img, species_label, geo, img_name = batch[:4]

        
        if use_cuda:
            img = img.cuda()
            species_label = species_label.cuda()
            geo = geo.cuda()

        if mode == "train":
            optimizer.zero_grad()
            pred = model(img)
        else:
            if len(img.shape) > 4:
                img = img.squeeze(0)
            bsz = img.shape[0]
            # print(bsz)
            
            img_batches = [
                img[args.batch_size * i: min(args.batch_size * (i+1), bsz), ...]
                for i in range((bsz + args.batch_size - 1)// args.batch_size)
            ]
            with torch.no_grad():
                pred = torch.cat([
                    model(img_batch).detach().clone()
                    for img_batch in img_batches
                ], 0)

        loss = loss_fn(pred, geo)

        if mode == "train":
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
            optimizer.step()
        else:
            if args.task_type == "audio_geoclip":
                pred = pred[..., -512:]
            metrics.update(
                pred.cpu(), geo.cpu(), img_name
            )
        avg_loss.append(loss.item())
        
        batch_pbar.update(1)
        batch_pbar.set_description("Batches {} | Loss: {:.4f}".format(mode, loss.item()))



    avg_loss = sum(avg_loss) / len(avg_loss) if len(avg_loss) else 0

    if mode == "train":
        return avg_loss
    else:
        metric_dict = metrics.get_metric_str()
        return avg_loss, metric_dict
