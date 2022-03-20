import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn

import models.cifar10 as cifar10_extra_models
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
    if args.hp.model_source == eppb.HyperParam.ModelSource.Local:
        model = cifar10_extra_models.__dict__[args.hp.arch](pretrained=args.hp.pretrained)
    else:
        raise NotImplementedError

    print("model:\n=========\n")
    display_model(model)

    process_model(model, args)

    # parallel and multi-gpu
    model = distributed_model(model, ngpus_per_node, args)

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = get_optimizer(model, args)
    cudnn.benchmark = True

    df = DataloaderFactory(args)
    train_loader, val_loader, train_sampler = df.product_train_val_loader(df.cifar10)
    writer = get_summary_writer(args, ngpus_per_node, model)
    if args.hp.evaluate:
        if writer is not None:
            get_model_info(model, args, val_loader)
    args.batch_num = len(train_loader)

    scheduler_lr = get_lr_scheduler(optimizer, args)

    if args.hp.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(0, args.start_epoch):
        scheduler_lr.step()
        pass
    for epoch in range(args.start_epoch, args.hp.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, writer)
        scheduler_lr.step()
        # evaluate on validation set
        acc1, acc5 = validate(val_loader, model, criterion, args)
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
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                prefix="{}/{}".format(args.log_name, args.arch),
            )
    if writer is not None:
        writer.close()
        if args.hp.send_wechat:
            send_wechat_info(
                title=args.hp.log_name,
                info="{}\n{:.2f}".format(args.hp.arch, best_acc1.item()),
            )


if __name__ == "__main__":
    main()
