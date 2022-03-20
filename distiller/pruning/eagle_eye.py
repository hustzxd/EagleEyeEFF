import queue
import random
import time
from enum import Enum
from functools import partial
from typing import Any, Dict, Iterable, List, Tuple, Type

import ipdb
import numpy as np
import torch
from black import FileContent
from rich.progress import track

import distiller


class SearchType(Enum):
    Evolution = 1
    Random = 2


def search_layer_spasity(
    model: torch.nn.Module,
    adjust_bn_fun,
    val_fun,
    logger,
    layer_to_prune: List,
    follower_layer_map,
    rank_type,
    log_dir,
    flops_ori,
    params_ori,
    dummy_input=torch.rand(1, 3, 32, 32),
    flops_pruned=0.5,
    best_queue_size=5,
    maxiter=100,
    search_type=SearchType.Random,
    step=0.02,
    min_sparsity=0.0,
    max_sparsity=0.8,
):

    cal_fitness = partial(
        _cal_fitness_base,
        adjust_bn_fun=adjust_bn_fun,
        val_fun=val_fun,
    )

    # ===========Ramdom Search>>>>>>>>>>>>>
    end = time.time()

    best_res = None
    best_fitness = 0
    increase = 0
    insert_first = True
    while insert_first:
        first = np.array([0.005 * increase for _ in range(len(layer_to_prune))])
        increase += 1
        pruned_model = distiller.prune_channel(
            model=model,
            layer_prune_name=layer_to_prune,
            per_layer_s=first,
            rank_type=rank_type,
            follower_layer_map=follower_layer_map,
        )
        flops, params, df = distiller.performance_summary(pruned_model, dummy_input)  # TODO
        # logger.debug("flops: {:.3f}M, params: {:.3f}M".format(flops / 10**6, params / 10**6))
        # logger.debug("\n{}\n".format(df))
        if flops_pruned - 0.005 <= 1 - flops / flops_ori:  # loop bug
            insert_first = False
            best_res = first
    best_fitness = cal_fitness(model=pruned_model)
    logger.info(
        "\npruned flops ratio: {:.3f} \nfitness: {:.3f} \nfirst average strategy: {}".format(
            1 - flops / flops_ori, best_fitness, ["{:.2f}".format(i) for i in best_res]
        )
    )
    best_fitness_queue = queue.PriorityQueue(maxsize=best_queue_size)
    if best_fitness_queue.full():
        best_fitness_queue.get()
    best_fitness_queue.put([best_fitness, -1, best_res])
    for i in range(maxiter):
        # step 1. generate code
        try_cnt = 0
        end_in = time.time()
        while True:
            try_cnt += 1
            # if (try_cnt > 10000): # do not check??
            #     logger.critical('can not search a stisfied strategy, please shrink step and retry.')
            if search_type == SearchType.Evolution:
                change_code = np.random.randint(3, size=len(layer_to_prune)) - 1  # \in {-1, 0, 1}
                random_idx = random.randint(0, best_fitness_queue.qsize() - 1)  # [0, size-1] random idx
                random_best_res = best_fitness_queue.queue[random_idx][2]
                res = random_best_res + change_code * step
                res = res.clip(min_sparsity, max_sparsity)
            elif search_type == SearchType.Random:
                res = _get_random_pruning_strategy(
                    len(layer_to_prune),
                    max_rate=max_sparsity,
                    min_rate=min_sparsity,
                )
            pruned_model = distiller.prune_channel(
                model=model,
                layer_prune_name=layer_to_prune,
                per_layer_s=res,
                rank_type=rank_type,
                follower_layer_map=follower_layer_map,
            )
            flops, params, df = distiller.performance_summary(pruned_model, dummy_input)  # TODO
            # logger.debug("\nflops: {:.3f}M, params: {:.3f}M".format(flops / 10**6, params / 10**6))
            # logger.debug("\n{}\n".format(df))
            if flops_pruned - 0.005 <= 1 - flops / flops_ori <= flops_pruned + 0.005:
                # legal pruning strategy
                break
        logger.info(
            "{}/{} search cost: {:6.3f}s".format(
                i,
                maxiter,
                time.time() - end_in,
            )
        )
        # step 2. cal the fitness
        fitness = cal_fitness(model=pruned_model)
        logger.info(
            "\npruned flops ratio: {:.3f} \nfitness: {:.3f} \nstrategy: {}".format(
                1 - flops / flops_ori, fitness, ["{:.2f}".format(i) for i in res]
            )
        )

        if best_fitness_queue.full():
            less_best = best_fitness_queue.get()
            if less_best[0] < fitness:
                logger.warning("{}: {:.2f} Update PriQueue".format(["{:.2f}".format(i) for i in less_best[2]], fitness))
                best_fitness_queue.put([fitness, i, res])
                with open("{}/best_queue.txt".format(log_dir), "w") as wf:
                    for i in range(best_fitness_queue.qsize()):
                        wf.write(
                            "v: {:.2f} cnt: {} k: {}\n".format(
                                best_fitness_queue.queue[i][0],
                                best_fitness_queue.queue[i][1],
                                best_fitness_queue.queue[i][2],
                            )
                        )
            else:
                best_fitness_queue.put(less_best)
        else:
            best_fitness_queue.put([fitness, i, res])

        # step 3. is best ?
        if fitness > best_fitness:
            best_fitness = fitness
            best_res = res
            logger.warning(
                "\nBest update: \nfitness:{:.2f} \nstrategy:{}".format(fitness, ["{:.2f}".format(i) for i in best_res])
            )
        # yes update best go step1
        # no, go step1
    logger.critical(
        "\nBest Fitness: {:.2f} \nBest Strategy: {} \nSearch cost:{} minutes".format(
            best_fitness, best_res, int((time.time() - end) / 60)
        )
    )

    while not best_fitness_queue.empty():
        fitness, cnt, res = best_fitness_queue.get()
        np.savetxt(
            "{}/prune{:.3f}_fitness{:.3f}.txt".format(log_dir, flops_pruned, fitness),
            res,
        )
    return best_res


def _cal_fitness_base(
    model,  # pruned_model
    adjust_bn_fun,
    val_fun,
):
    adjust_bn_fun(model)
    # evaluate on validation set
    acc1, _ = val_fun(model)
    # logger.info("{}: {:.2f}".format(["{:.2f}".format(i) for i in sparsity_config], acc1.item()))
    return acc1.item()


def _get_random_pruning_strategy(num_layer, max_rate, min_rate):
    channel_config = np.random.rand(num_layer)
    channel_config = channel_config * (max_rate - min_rate) + min_rate
    return channel_config
