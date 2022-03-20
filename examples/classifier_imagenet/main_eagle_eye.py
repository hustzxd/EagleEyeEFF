from functools import partial

import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn

import models.imagenet as imagenet_extra_models
from examples import *

best_acc1 = 0


def main():
    parser = get_base_parser()
    args = parser.parse_args()
    hp = get_hyperparam(args)
    if hp.gpu_id == eppb.GPU.ANY:
        args.gpu = get_freer_gpu()
    elif hp.gpu_id == eppb.GPU.NONE:
        args.gpu = None  # TODO: test

    main_s1_set_seed(hp)
    main_s2_start_worker(main_worker, args, hp)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    args.hp = get_hyperparam(args)

    if args.distributed:
        if args.hp.multi_gpu.dist_url == "env://" and args.hp.multi_gpu.rank == -1:
            args.hp.multi_gpu.rank = int(os.environ["RANK"])
        if args.hp.multi_gpu.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.hp.multi_gpu.rank = args.hp.multi_gpu.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.hp.multi_gpu.dist_backend,
            init_method=args.hp.multi_gpu.dist_url,
            world_size=args.world_size,
            rank=args.hp.multi_gpu.rank,
        )
    # create model
    if args.hp.pretrained:
        print("=> using pre-trained model '{}'".format(args.hp.arch))
    else:
        print("=> creating model '{}'".format(args.hp.arch))
    if args.hp.model_source == eppb.HyperParam.ModelSource.TorchVision:
        model = torchvision.models.__dict__[args.hp.arch](pretrained=args.hp.pretrained)
    elif args.hp.model_source == eppb.HyperParam.ModelSource.PyTorchCV:
        model = ptcv_get_model(args.hp.arch, pretrained=args.hp.pretrained)
    elif args.hp.model_source == eppb.HyperParam.ModelSource.Local:
        model = imagenet_extra_models.__dict__[args.hp.arch](pretrained=args.hp.pretrained)

    print("model:\n=========\n")
    display_model(model)

    process_model(model, args)

    df = DataloaderFactory(args)
    train_loader, val_loader, train_sampler = df.product_train_val_loader(df.imagenet2012)
    writer = get_summary_writer(args, ngpus_per_node, model)
    logger = get_color_logger(writer, args)

    args.batch_num = len(train_loader)

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    distributed_model(model, ngpus_per_node, args)

    flops_ori, params_ori, df_ori = distiller.performance_summary(model, dummy_input=torch.rand(1, 3, 224, 224))  # TODO
    logger.info("\n{}\n".format(df_ori))
    logger.info("\noriginal model flops: {:.3f}M, params: {:.3f}M".format(flops_ori / 10**6, params_ori / 10**6))

    pruned_model = None
    best_per_layer_s = None

    if args.hp.rcp.HasField("freeze_sparsity"):
        assert not args.hp.rcp.HasField("ada")
        logger.info("Use freezed sparsity")
        best_per_layer_s = np.loadtxt(args.hp.rcp.freeze_sparsity)
        layer_name_list, _, follower_layer_map = prepare_replace_map(args.hp.replace_layer_map)
        assert len(layer_name_list) == len(best_per_layer_s), "{}!={}".format(
            len(layer_name_list), len(best_per_layer_s)
        )
        pruned_model = distiller.prune_channel(
            model=model,
            layer_prune_name=layer_name_list,
            per_layer_s=best_per_layer_s,
            rank_type=str_rank_type_map[args.hp.rcp.rank_type],
            follower_layer_map=follower_layer_map,
        )
        pruned_flops, pruned_params, pruned_df = distiller.performance_summary(
            pruned_model, dummy_input=torch.rand(1, 3, 224, 224)
        )
        p_ratio = pruned_flops / flops_ori
        logger.info("\n{}\n".format(pruned_df))
        logger.info(
            "Use freezed sparsity, and the left flops ratio: {:.3f}, left params ratio: {:.3f}".format(
                p_ratio, pruned_params / params_ori
            )
        )

    if args.hp.rcp.HasField("ada"):  # differential_evolution
        assert not args.hp.rcp.HasField("freeze_sparsity")
        layer_name_list, _, follower_layer_map = prepare_replace_map(args.hp.replace_layer_map)
        adjust_bn_fun = partial(adjust_bn, train_loader=train_loader, args=args)
        val_fun = partial(fast_val, val_loader=val_loader, args=args)
        best_per_layer_s = distiller.eagle_eye.search_layer_spasity(
            model=model,
            adjust_bn_fun=adjust_bn_fun,
            val_fun=val_fun,
            logger=logger,
            log_dir=args.log_name,
            layer_to_prune=layer_name_list,
            follower_layer_map=follower_layer_map,
            flops_ori=flops_ori,
            params_ori=params_ori,
            rank_type=str_rank_type_map[args.hp.rcp.rank_type],
            dummy_input=torch.randn(1, 3, 224, 224),
            flops_pruned=args.hp.rcp.flops_pruned,
            best_queue_size=args.hp.rcp.ada.best_queue_size,
            search_type=str_search_type_map[args.hp.rcp.ada.search_type],
            maxiter=args.hp.rcp.ada.maxiter,
            step=args.hp.rcp.ada.step,
            min_sparsity=args.hp.rcp.ada.min_sparsity,
            max_sparsity=args.hp.rcp.ada.max_sparsity,
        )
        pruned_model = distiller.prune_channel(
            model=model,
            layer_prune_name=layer_name_list,
            per_layer_s=best_per_layer_s,
            rank_type=str_rank_type_map[args.hp.rcp.rank_type],
            follower_layer_map=follower_layer_map,
        )
        pruned_flops, pruned_params, df = distiller.performance_summary(
            pruned_model, dummy_input=torch.rand(1, 3, 224, 224)
        )
        p_ratio = pruned_flops / flops_ori
        logger.info("\n{}\n".format(df))
        logger.info(
            "\nthe left flops ratio: {:.3f} \n the left para ratio: {:.3f}".format(p_ratio, pruned_params / params_ori)
        )
        # <<<<<<<<<<<<Ramdom Search===============

    del model
    # parallel and multi-gpu
    pruned_model = distributed_model(pruned_model, ngpus_per_node, args)

    optimizer = get_optimizer(pruned_model, args)
    cudnn.benchmark = True
    scheduler_lr = get_lr_scheduler(optimizer, args)

    adjust_bn(pruned_model, train_loader=train_loader, args=args, early_exit=100)
    # evaluate on validation set
    acc1, _ = validate(model=pruned_model, val_loader=val_loader, criterion=criterion, args=args)
    logger.info(
        "\nFitness: {:.2f} \nPruning Strategy: {}".format(acc1.item(), ["{:.2f}".format(i) for i in best_per_layer_s])
    )

    for epoch in range(0, args.hp.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # train for one epoch
        train(train_loader, pruned_model, criterion, optimizer, epoch, args, writer)
        scheduler_lr.step()
        # evaluate on validation set
        acc1, acc5 = validate(val_loader=val_loader, model=pruned_model, criterion=criterion, args=args)
        if writer is not None:
            writer.add_scalar("val/acc1", acc1, epoch)
            writer.add_scalar("val/acc5", acc5, epoch)
            writer.add_scalar("val/lr", optimizer.param_groups[0]["lr"], epoch)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if writer is not None:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": pruned_model.state_dict(),
                    "best_acc1": best_acc1,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                prefix="{}/{}".format(args.log_name, args.arch),
            )
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
