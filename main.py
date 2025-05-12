import src
from src import dataset, models, utils, train_eval, params, config


import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import logging
import os
import copy
import json
import numpy as np
import random

def run_train(args):
    weight_dir, plot_dir, pred_dir = utils.setup_logging(args, mode="train")

    use_cuda = torch.cuda.is_available()
    
    train_dataloader, val_dataloader, test_dataloader = dataset.get_dataloaders(args)
    num_classes = train_dataloader.num_classes
    
    logging.info("Number of classes: " + str(num_classes))
    logging.info("Number of geo-classes: " + str(train_dataloader.num_geo_classes))
        
    output_dim = train_dataloader.num_geo_classes
    model = models.get_model(
        args.model, 
        output_dim=output_dim, 
        pretrained=args.pretrained, 
        dropout=args.dropout,
        args=args, 
    )
    # model = nn.DataParallel(model)
    assert args.model_weight == "" or args.encoder_weight == ""
    if args.model_weight != "":
        weights = torch.load(args.model_weight)
        model.load_state_dict(weights, strict=False)
    if args.encoder_weight != "":
        weights = torch.load(args.encoder_weight)
        weights = {k:v for k, v in weights.items() if not ("fc" in k or "heads" in k or "classifier.3" in k)}
        model.load_state_dict(weights, strict=False)
    
    print("model loaded")
    model.train()
    if use_cuda:
        model = model.cuda()

    optimizer = models.get_opt(args, model)

    BASE_EPOCHS = 50
    # BASE_EPOCHS = args.epochs
    # Warmup for 5 epochs and then cosine decay for BASE_EPOCHS - 5 epochs and then step decay for the rest
    scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=BASE_EPOCHS - 5, eta_min=0.1*args.lr)
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[5])


    loss_fn = models.get_loss(args)    
    print(model)
    num_param = sum(p.numel() for p in model.parameters())
    num_param2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters: ", num_param, num_param2)
    logging.info("Number of parameters: " + str(num_param) + " " + str(num_param2))
    
    logging.info(str(model))
    logging.info(str(args))

    epoch2loss = {"train": {}, "val": {}}
    epoch2lr = {"train": {}}
    epoch_pbar = tqdm.tqdm(total=args.epochs, desc="epoch")
    LOG_FMT = "Epoch {:3d} | {} set | LR {:.4E} | Loss {:.4f}"
    
    best_val_metrics = None
    best_weights = None
    best_epoch = None
    
    for epoch in range(args.epochs):

        cur_lr = lr_scheduler.get_last_lr()[-1]
        epoch2lr["train"][epoch] = cur_lr


        ## Train set
        train_loss = train_eval.run_loop(
            args, train_dataloader, model, loss_fn,
            mode="train", optimizer=optimizer, 
            use_cuda=use_cuda, 
            epoch=epoch,
        )
        logging.info(LOG_FMT.format(
            epoch, "train", cur_lr, 
            train_loss, 
        ))
        epoch2loss["train"][epoch] = train_loss
        
        
        if (epoch+1) % args.save_freq == 0:
            utils.save_model(model, os.path.join(weight_dir, "checkpoint_{}.pt".format(epoch)))
        if (epoch+1) % args.eval_freq == 0:
            ## Val set
            val_loss, val_metrics = train_eval.run_loop(
                args, val_dataloader, model, loss_fn,
                mode="eval",
                use_cuda=use_cuda,
                save_dir=os.path.join(pred_dir, "val", "epoch_{}".format(epoch))
            )
            logging.info(LOG_FMT.format(
                epoch, "val", cur_lr, 
                val_loss, 
            ))
            logging.info(val_metrics)
            epoch2loss["val"][epoch] = val_loss

            if True: #best_val_metrics is None or utils.metric_str2acc(best_val_metrics) < utils.metric_str2acc(val_metrics):
                best_val_metrics = val_metrics
                best_weights = copy.deepcopy(model.state_dict())
                best_epoch = epoch

        lr_scheduler.step()

        epoch_pbar.update(1)
        epoch_pbar.set_description("Epochs | LR: {:.4E} Loss: {:.4f}".format(cur_lr, train_loss))


        ## Plots
        utils.save_plots(epoch2loss, os.path.join(plot_dir, "loss.png"))
        utils.save_plots(epoch2lr, os.path.join(plot_dir, "lr.png"))
    
    logging.info("Best Val: Epoch {}, Best metrics".format(best_epoch))
    logging.info(best_val_metrics)
    model.load_state_dict(best_weights)
    ## Test set
    if test_dataloader is None:
        return
    test_loss, test_metrics = train_eval.run_loop(
        args, test_dataloader, model, loss_fn,
        mode="eval",
        use_cuda=use_cuda,
        save_dir=os.path.join(pred_dir, "test", "epoch_{}".format(epoch))
    )
    logging.info(LOG_FMT.format(
        best_epoch, "test", cur_lr, 
        test_loss, 
    ))
    logging.info(test_metrics)



def run_eval(args):
    weight_dir, plot_dir, pred_dir = utils.setup_logging(args, mode="eval")

    use_cuda = torch.cuda.is_available()
    train_dataloader, val_dataloader, test_dataloader = dataset.get_dataloaders(args)
    num_classes = train_dataloader.num_classes

    output_dim = train_dataloader.num_geo_classes
    model = models.get_model(
        args.model, 
        output_dim=output_dim, 
        pretrained=args.pretrained,
        args=args, 
    )
    loss_fn = models.get_loss(args)  
    if args.model_weight != "":
        weights = torch.load(args.model_weight)
        model.load_state_dict(weights, strict=False)
    assert args.encoder_weight == ""
    print("model loaded")
    # model = nn.DataParallel(model)
    model.eval()
    if use_cuda:
        model = model.cuda()
    
    num_param = sum(p.numel() for p in model.parameters())
    num_param2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters: ", num_param, num_param2)
    logging.info("Number of parameters: " + str(num_param) + " " + str(num_param2))
    
    logging.info(str(model))
    logging.info(str(args))

    LOG_FMT = "{} set | Loss {:.4f}"
        
    val_loss, val_sound_metrics = train_eval.run_loop(
        args, val_dataloader, model, loss_fn,
        mode="eval",
        use_cuda=use_cuda,
        save_dir=os.path.join(pred_dir, "val"),
    )
    logging.info(LOG_FMT.format(
        "val", 
        val_loss, 
    ))
    logging.info(val_sound_metrics)
    print(val_sound_metrics)

    if test_dataloader is None or args.val_only:
        return
    test_loss, test_sound_metrics = train_eval.run_loop(
        args, test_dataloader, model, loss_fn,
        mode="eval",
        use_cuda=use_cuda,
        save_dir=os.path.join(pred_dir, "test"),
    )
    logging.info(LOG_FMT.format(
        "test", 
        test_loss, 
    ))
    logging.info(test_sound_metrics)
    print(test_sound_metrics)


if __name__=="__main__":
    args = params.get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # torch.autograd.set_detect_anomaly(True)
    if args.mode == "train":
        run_train(args)
    elif args.mode == "eval":
        run_eval(args)