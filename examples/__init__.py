import argparse
import datetime
import hashlib
import logging
import os
import random
import shutil
import time
import warnings

import google.protobuf as pb
import google.protobuf.text_format
import numpy as np
import plotly.graph_objects as go
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
from pytorchcv.model_provider import get_model as ptcv_get_model
from tensorboardX import SummaryWriter
from warmup_scheduler import GradualWarmupScheduler

import distiller
import models._modules as my_nn
from proto import efficient_pytorch_pb2 as eppb

str_rank_type_map = {
    eppb.RCP.RankType.L1Norm: distiller.RankType.L1Norm,
    eppb.RCP.RankType.Random: distiller.RankType.Random,
}

str_search_type_map = {
    eppb.RCP.AdaParam.SearchType.Evolution: distiller.eagle_eye.SearchType.Evolution,
    eppb.RCP.AdaParam.SearchType.Random: distiller.eagle_eye.SearchType.Random,
}


def get_base_parser():
    """
    Default values should keep stable.
    """

    print("Please do not import ipdb when using distributed training")

    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument("--hp", type=str, help="File path to save hyperparameter configuration")
    parser.add_argument("--nbits-a", type=int, default=-1, help="Override the hp.nbits_a")
    parser.add_argument("--arch", type=str, default="None", help="Override the args.hp.arch")

    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "--resume-after",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )

    parser.add_argument("--bn-fusion", action="store_true", default=False, help="ConvQ + BN fusion")
    parser.add_argument("--resave", action="store_true", default=False, help="resave the model")

    parser.add_argument(
        "--gen-layer-info",
        action="store_true",
        default=False,
        help="whether to generate layer information for latency evaluation on hardware",
    )

    parser.add_argument(
        "--print-histogram",
        action="store_true",
        default=False,
        help="save histogram of weight in tensorboard",
    )
    parser.add_argument("--freeze-bn", action="store_true", default=False, help="Freeze BN")
    return parser


def main_s1_set_seed(hp):
    if hp.HasField("seed"):
        random.seed(hp.seed)
        torch.manual_seed(hp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )


def main_s2_start_worker(main_worker, args, hp):
    if args.gpu is not None:
        warnings.warn("You have chosen a specific GPU. This will completely " "disable data parallelism.")
    args.world_size = hp.multi_gpu.world_size
    if hp.HasField("multi_gpu") and hp.multi_gpu.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or (hp.HasField("multi_gpu") and hp.multi_gpu.multiprocessing_distributed)

    ngpus_per_node = torch.cuda.device_count()
    print("ngpus_per_node: {}".format(ngpus_per_node))
    if hp.HasField("multi_gpu") and hp.multi_gpu.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def get_hyperparam(args):
    assert os.path.exists(args.hp)
    hp = eppb.HyperParam()
    with open(args.hp, "r") as rf:
        pb.text_format.Merge(rf.read(), hp)
    return hp


def get_freer_gpu():
    # A5 5lines for NVIDIA-SMI 510.47.03    Driver Version: 510.47.03    CUDA Version: 11.6
    os.system("nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp")
    memory_available = [int(x.split()[2]) for x in open("tmp", "r").readlines()]
    os.system("rm tmp")
    # TODO; if no gpu, return None
    try:
        visible_gpu = os.environ["CUDA_VISIBLE_DEVICES"]
        memory_visible = []
        for i in visible_gpu.split(","):
            memory_visible.append(memory_available[int(i)])
        return np.argmax(memory_visible)
    except KeyError:
        return np.argmax(memory_available)


def get_lr_scheduler(optimizer, lr_domain):
    """
    Args:
        optimizer:
        lr_domain ([proto]): [lr configuration domain] e.g. args.hp args.hp.bit_pruner
    """
    if isinstance(lr_domain, argparse.Namespace):
        lr_domain = lr_domain.hp
    if lr_domain.lr_scheduler == eppb.LRScheduleType.CosineAnnealingLR:
        print("Use cosine scheduler")
        scheduler_next = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=lr_domain.epochs)
    elif lr_domain.lr_scheduler == eppb.LRScheduleType.StepLR:
        print(
            "Use step scheduler, step size: {}, gamma: {}".format(lr_domain.step_lr.step_size, lr_domain.step_lr.gamma)
        )
        scheduler_next = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=lr_domain.step_lr.step_size,
            gamma=lr_domain.step_lr.gamma,
        )
    elif lr_domain.lr_scheduler == eppb.LRScheduleType.MultiStepLR:
        print(
            "Use MultiStepLR scheduler, milestones: {}, gamma: {}".format(
                lr_domain.multi_step_lr.milestones, lr_domain.multi_step_lr.gamma
            )
        )
        scheduler_next = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=lr_domain.multi_step_lr.milestones,
            gamma=lr_domain.multi_step_lr.gamma,
        )
    elif lr_domain.lr_scheduler == eppb.LRScheduleType.CyclicLR:
        print("Use CyclicLR scheduler")
        if not lr_domain.cyclic_lr.HasField("step_size_down"):
            step_size_down = None
        else:
            step_size_down = lr_domain.cyclic_lr.step_size_down

        cyclic_mode_map = {
            eppb.CyclicLRParam.Mode.triangular: "triangular",
            eppb.CyclicLRParam.Mode.triangular2: "triangular2",
            eppb.CyclicLRParam.Mode.exp_range: "exp_range",
        }

        scheduler_next = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=lr_domain.cyclic_lr.base_lr,
            max_lr=lr_domain.cyclic_lr.max_lr,
            step_size_up=lr_domain.cyclic_lr.step_size_up,
            step_size_down=step_size_down,
            mode=cyclic_mode_map[lr_domain.cyclic_lr.mode],
            gamma=lr_domain.cyclic_lr.gamma,
        )
    else:
        raise NotImplementedError
    if not lr_domain.HasField("warmup"):
        return scheduler_next
    print("Use warmup scheduler")
    lr_scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=lr_domain.warmup.multiplier,
        total_epoch=lr_domain.warmup.epochs,
        after_scheduler=scheduler_next,
    )
    return lr_scheduler


def get_optimizer(model, args):
    # define optimizer after process model
    print("define optimizer")
    if args.hp.optimizer == eppb.OptimizerType.SGD:
        params = add_weight_decay(
            model,
            weight_decay=args.hp.sgd.weight_decay,
            skip_keys=[
                "expand_",
                "running_scale",
                "alpha",
                "standard_threshold",
                "nbits",
            ],
        )
        optimizer = torch.optim.SGD(params, args.hp.lr, momentum=args.hp.sgd.momentum)
        print("Use SGD")
    elif args.hp.optimizer == eppb.OptimizerType.Adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.hp.lr, weight_decay=args.hp.adam.weight_decay)
        print("Use Adam")
    else:
        raise NotImplementedError
    return optimizer


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def add_weight_decay(model, weight_decay, skip_keys):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        added = False
        for skip_key in skip_keys:
            if skip_key in name:
                no_decay.append(param)
                added = True
                break
        if not added:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def validate(model, val_loader, criterion, args):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5, prefix="Test: ")

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.hp.print_freq == 0:
                progress.print(i)
            if args.hp.overfit_test:
                break

        print(
            " *Time {time.sum:.0f}s Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(
                time=batch_time, top1=top1, top5=top5
            )
        )

    return top1.avg, top5.avg


def fast_val(model, val_loader, args):
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if i % 5 != 0:  # shrink the val size to 1/5
                continue
            if args.gpu is not None:
                input = input.cuda(args.gpu)
                target = target.cuda(args.gpu)
            # compute output
            output = model(input)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))
    return top1.avg, top5.avg


def adjust_bn(model, train_loader, args, early_exit=100):
    # =========
    model.eval()  # adaptive BN
    model.apply(distiller.set_bn_train)  # adaptive BN
    model.apply(distiller.reset_bn)  # adaptive BN
    # <<<<<<<<<<
    # train for one epoch
    for i, data in enumerate(train_loader):  # for imagenet, we need shrink the iter
        # measure data loading time
        inputs = data[0]
        targets = data[1]
        if args.gpu is not None:
            inputs = inputs.cuda(args.gpu)
            targets = targets.cuda(args.gpu)
        # compute output
        _ = model(inputs)
        if i > early_exit:  # early quit
            break


def train(train_loader, model, criterion, optimizer, epoch, args, writer, logger=None):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        args.batch_num,
        batch_time,
        data_time,
        losses,
        top1,
        top5,
        prefix="Epoch: [{}]".format(epoch),
        logger=logger,
    )
    print("gpu id: {}".format(args.gpu))
    # switch to train mode
    model.train()

    end = time.time()
    base_step = epoch * args.batch_num
    for i, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs = data[0]
        targets = data[1]
        if args.gpu is not None:
            inputs = inputs.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)
        # compute output
        output = model(inputs)
        loss = criterion(output, targets)
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))
        if writer is not None:
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], base_step + i)
            writer.add_scalar("train/acc1", top1.avg, base_step + i)
            writer.add_scalar("train/acc5", top5.avg, base_step + i)
        # compute gradient and do SGD step
        loss.backward()
        # optimizer.param_groups[1]['params'][3].grad
        optimizer.step()
        optimizer.zero_grad()
        # warning 1. backward 2. step 3. zero_grad
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.hp.print_freq == 0:
            progress.print(i)
            if writer is not None and args.hp.debug:
                pass
        if args.hp.overfit_test:
            break
    return


def get_summary_writer(args, ngpus_per_node, model):
    if not args.hp.multi_gpu.multiprocessing_distributed or (
        args.hp.multi_gpu.multiprocessing_distributed and args.hp.multi_gpu.rank % ngpus_per_node == 0
    ):
        args.log_name = "logger/{}_{}_{}".format(args.hp.arch, args.hp.log_name, get_current_time())
        writer = SummaryWriter(args.log_name)
        with open("{}/{}.prototxt".format(args.log_name, args.arch), "w") as wf:
            wf.write(str(args.hp))
        with open("{}/{}.txt".format(args.log_name, args.arch), "w") as wf:
            wf.write(str(model))
        return writer
    return None


def get_color_logger(writer, args):
    if writer is None:
        return None
    if args.hp.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger = distiller.log.get_color_logger(filename="{}/log.txt".format(args.log_name), level=level)
    return logger


def save_checkpoint(state, is_best, prefix, filename="checkpoint.pth.tar"):
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + "best.pth.tar")
    return


def process_model(model, args):
    if (not hasattr(args, "arch")) or args.arch == "None":
        args.arch = args.hp.arch

    if args.hp.HasField("weight"):
        if os.path.isfile(args.hp.weight):
            print("=> loading weight '{}'".format(args.hp.weight))
            weight = torch.load(args.hp.weight, map_location="cpu")
            model.load_state_dict(weight)
        else:
            print("=> no weight found at '{}'".format(args.hp.weight))

    if args.hp.HasField("resume"):
        if os.path.isfile(args.hp.resume):
            print("=> loading checkpoint '{}'".format(args.hp.resume))
            checkpoint = torch.load(args.hp.resume, map_location="cpu")
            try:
                model.load_state_dict(checkpoint["state_dict"])
            except RuntimeError as e:
                print(e)
            print("best_acc: {}".format(checkpoint["best_acc1"].item()))
        else:
            print("=> no checkpoint found at '{}'".format(args.hp.resume))
    return


class DataloaderFactory(object):
    # MNIST
    mnist = 0
    # CIFAR10
    cifar10 = 10
    cifar10_split = 11
    # ImageNet2012
    imagenet2012 = 40
    imagenet2012_split = 42

    def __init__(self, args):
        self.args = args
        self.mnist_transform = transforms.Compose(
            [
                transforms.Resize([32, 32]),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        self.cifar10_transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        self.cifar10_transform_val = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

    def product_train_val_loader(self, data_type):
        args = self.args
        noverfit = not args.hp.overfit_test
        train_loader = None
        val_loader = None
        # MNIST
        if data_type == self.mnist:
            train_loader = torch.utils.data.DataLoader(
                torchvision.datasets.MNIST(
                    args.hp.data,
                    train=True,
                    download=True,
                    transform=self.mnist_transform,
                ),
                batch_size=args.hp.batch_size,
                shuffle=True and noverfit,
            )
            val_loader = torch.utils.data.DataLoader(
                torchvision.datasets.MNIST(args.hp.data, train=False, transform=self.mnist_transform),
                batch_size=args.hp.batch_size,
                shuffle=False,
            )
            return train_loader, val_loader
        # CIFAR10
        if data_type == self.cifar10:
            trainset = torchvision.datasets.CIFAR10(
                root=args.hp.data,
                train=True,
                download=True,
                transform=self.cifar10_transform_train,
            )
            if args.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
            else:
                train_sampler = None
            train_loader = torch.utils.data.DataLoader(
                trainset,
                batch_size=args.hp.batch_size,
                shuffle=(train_sampler is None) and noverfit,
                num_workers=args.hp.workers,
                sampler=train_sampler,
            )
            testset = torchvision.datasets.CIFAR10(
                root=args.hp.data,
                train=False,
                download=True,
                transform=self.cifar10_transform_val,
            )
            val_loader = torch.utils.data.DataLoader(
                testset,
                batch_size=args.hp.batch_size,
                shuffle=False,
                num_workers=args.hp.workers,
            )
            return train_loader, val_loader, train_sampler

        # ImageNet
        elif data_type == self.imagenet2012:
            # Data loading code
            traindir = os.path.join(args.hp.data, "train")
            valdir = os.path.join(args.hp.data, "val")
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            train_dataset = torchvision.datasets.ImageFolder(
                traindir,
                transforms.Compose(
                    [
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            )

            if args.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            else:
                train_sampler = None

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.hp.batch_size,
                shuffle=(train_sampler is None) and noverfit,
                num_workers=args.hp.workers,
                pin_memory=True,
                sampler=train_sampler,
            )

            val_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(
                    valdir,
                    transforms.Compose(
                        [
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize,
                        ]
                    ),
                ),
                batch_size=args.hp.batch_size,
                shuffle=False,
                num_workers=args.hp.workers,
                pin_memory=True,
            )
            return train_loader, val_loader, train_sampler
        else:
            assert NotImplementedError
        return train_loader, val_loader


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix="", logger=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if self.logger is None:
            print("\t".join(entries))
        else:
            self.logger.info("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0
        self.name = name
        self.fmt = fmt

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def distributed_model(model, ngpus_per_node, args):
    if not torch.cuda.is_available() or args.gpu is None:
        print("using CPU, this will be slow")
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(int(args.gpu))
            model.cuda(args.gpu)

            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.hp.batch_size = int(args.hp.batch_size / ngpus_per_node)
            args.hp.workers = int((args.hp.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            assert NotImplementedError
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(int(args.gpu))
        model = model.cuda(args.gpu)
    else:
        assert NotImplementedError
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.hp.arch.startswith("alexnet") or args.hp.arch.startswith("vgg"):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    return model


def get_hash_code(message):
    hash = hashlib.sha1(message.encode("UTF-8")).hexdigest()
    return hash[:8]


def get_current_time():
    currentDT = datetime.datetime.now()
    return currentDT.strftime("%Y-%m-%d-%H:%M")


def display_model(model):
    str_list = str(model).split("\n")
    if len(str_list) < 30:
        print(model)
        return
    begin = 10
    end = 5
    middle = len(str_list) - begin - end
    abbr_middle = ["...", "{} lines".format(middle), "..."]
    abbr_str = "\n".join(str_list[:10] + abbr_middle + str_list[-5:])
    print(abbr_str)


def def_module_name(model):
    for module_name, module in model.named_modules():
        module.__name__ = module_name


def prepare_replace_map(replace_map):
    layer_name_list = []
    layer_type_list = []
    # layer_param_list = [] # TODO
    follower_layer_map = {}
    for layer_map in replace_map.layer:
        layer_name_list.append(layer_map.name[0])
        for fow in layer_map.name[1:]:
            follower_layer_map[fow] = layer_map.name[0]
        layer_type_list.append(layer_map.type)
    return layer_name_list, layer_type_list, follower_layer_map
