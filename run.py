import argparse
from logging import getLogger
import os
import sys
from collections.abc import MutableMapping

from recbole.config import Config
from recbole_gnn.config import Config as Config_gnn
from recbole.data import create_dataset, data_preparation
from recbole.data.utils import get_dataloader
from recbole.utils import init_logger, init_seed, get_model, get_trainer, set_color
from recbole_gnn.utils import (
    create_dataset as create_dataset_gnn,
    data_preparation as data_preparation_gnn,
    get_model as get_model_gnn,
    get_trainer as get_trainer_gnn
)

from recbole.data.dataloader import FullSortEvalDataLoader
from recbole.data.interaction import Interaction

import numpy as np
import pandas as pd
import pickle
import glob
from preprocess import create_session_graph, detect_communities, add_edges

from models.clip_gru4rec import CLIP_GRU4Rec
from models.clip_narm import CLIP_NARM
from models.clip_core import CLIP_CORE
from models.clip_srgnn import CLIP_SRGNN
from models.clip_tagnn import CLIP_TAGNN
from models.clip_gcsan import CLIP_GCSAN
from models.clip_gcegnn import CLIP_GCEGNN
from models.clip_lessr import CLIP_LESSR

def run_session(
    model,
    dataset,
    config_file_list=None,
    config_dict=None,
    saved=True,
    nproc=1,
    world_size=-1,
    ip="localhost",
    port="5678",
    group_offset=0,
):
    if nproc == 1 and world_size <= 0:
        res = run_rec(
            model=model,
            dataset=dataset,
            config_file_list=config_file_list,
            config_dict=config_dict,
            saved=saved,
        )
    else:
        if world_size == -1:
            world_size = nproc
        import torch.multiprocessing as mp

        # Refer to https://discuss.pytorch.org/t/problems-with-torch-multiprocess-spawn-and-simplequeue/69674/2
        # https://discuss.pytorch.org/t/return-from-mp-spawn/94302/2
        queue = mp.get_context("spawn").SimpleQueue()

        config_dict = config_dict or {}
        config_dict.update(
            {
                "world_size": world_size,
                "ip": ip,
                "port": port,
                "nproc": nproc,
                "offset": group_offset,
            }
        )
        kwargs = {
            "config_dict": config_dict,
            "queue": queue,
        }

        mp.spawn(
            run_recs,
            args=(model, dataset, config_file_list, kwargs),
            nprocs=nproc,
            join=True,
        )

        # Normally, there should be only one item in the queue
        res = None if queue.empty() else queue.get()
    return res

def run_rec(
    model=None,
    dataset=None,
    config_file_list=None,
    config_dict=None,
    saved=True,
    queue=None,
):
    if model == 'CLIP_GRU4Rec':
        config = Config(model=CLIP_GRU4Rec, dataset=dataset, config_file_list=[f'properties/{dataset}.yaml', f'properties/models/CLIP_GRU4Rec.yaml'], config_dict=config_dict)
    elif model == 'CLIP_NARM':
        config = Config(model=CLIP_NARM, dataset=dataset, config_file_list=[f'properties/{dataset}.yaml', f'properties/models/CLIP_NARM.yaml'], config_dict=config_dict)
    elif model == 'CLIP_CORE':
        config = Config(model=CLIP_CORE, dataset=dataset, config_file_list=[f'properties/{dataset}.yaml', f'properties/models/CLIP_CORE.yaml'], config_dict=config_dict)
    elif model == 'CLIP_SRGNN':
        config = Config(model=CLIP_SRGNN, dataset=dataset, config_file_list=[f'properties/{dataset}.yaml', f'properties/models/CLIP_SRGNN.yaml'], config_dict=config_dict)
    elif model == 'CLIP_TAGNN':
        config = Config_gnn(model=CLIP_TAGNN, dataset=dataset, config_file_list=[f'properties/{dataset}.yaml', 'properties/models/CLIP_TAGNN.yaml'], config_dict=config_dict)
    elif model == 'CLIP_GCSAN':
        config = Config(model=CLIP_GCSAN, dataset=dataset, config_file_list=[f'properties/{dataset}.yaml', f'properties/models/CLIP_GCSAN.yaml'], config_dict=config_dict)
    elif model == 'CLIP_GCEGNN':
        config = Config_gnn(model=CLIP_GCEGNN, dataset=dataset, config_file_list=[f'properties/{dataset}.yaml', 'properties/models/CLIP_GCEGNN.yaml'], config_dict=config_dict)
    elif model == 'CLIP_LESSR':
        config = Config_gnn(model=CLIP_LESSR, dataset=dataset, config_file_list=[f'properties/{dataset}.yaml', 'properties/models/CLIP_LESSR.yaml'], config_dict=config_dict)
    elif model in ['TAGNN','GCEGNN','LESSR']:
        config = Config_gnn(model=model, dataset=dataset, config_file_list=[f'properties/{dataset}.yaml', f'properties/models/{model}.yaml'], config_dict=config_dict)
    else:
        config = Config(model=model, dataset=dataset, config_file_list=[f'properties/{dataset}.yaml'], config_dict=config_dict)

    init_seed(config["seed"], config["reproducibility"])
    config['train_batch_size'] = config['batch_size']
    config['eval_batch_size'] = config['batch_size']
    
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)
    
    if model in ['TAGNN','CLIP_TAGNN','GCEGNN','CLIP_GCEGNN','LESSR','CLIP_LESSR']:
        dataset = create_dataset_gnn(config)
        train_data, valid_data, test_data = data_preparation_gnn(config, dataset)
    else:
        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)
    
    logger.info(dataset)

    # train_data, valid_data, test_data = data_preparation(config, dataset)

    if 'CLIP_' in config['model']:
        partition_file_path = config['data_path'] + '/partition.pickle'

        if not os.path.exists(partition_file_path):
            G = create_session_graph(train_data)
            G = add_edges(G, valid_data)
            G = add_edges(G, test_data)
        
            resolution = 1
            partition = detect_communities(G, resolution)

            with open(partition_file_path, 'wb') as f:
                pickle.dump(partition, f)
        else:
            print(f"Partition file {partition_file_path} already exists. Skipping community detection.")
        
        with open(partition_file_path, 'rb') as f:
            partition = pickle.load(f)

        config['n_community'] = len(set(partition.values()))

    # custom data - recbole gnn
    if config["model"] == 'CLIP_TAGNN':
        config["model"] = 'TAGNN'
        dataset = create_dataset_gnn(config)
        train_data, valid_data, test_data = data_preparation_gnn(config, dataset)
        config["model"] = 'CLIP_TAGNN'
    elif config["model"] == 'CLIP_GCEGNN':
        config["model"] = 'GCEGNN'
        dataset = create_dataset_gnn(config)
        train_data, valid_data, test_data = data_preparation_gnn(config, dataset)
        config["model"] = 'CLIP_GCEGNN'
    elif config["model"] == 'CLIP_LESSR':
        config["model"] = 'LESSR'
        dataset = create_dataset_gnn(config)
        train_data, valid_data, test_data = data_preparation_gnn(config, dataset)
        config["model"] = 'CLIP_LESSR'

    # model loading and initialization
    if config["model"] == 'CLIP_GRU4Rec':
        model = CLIP_GRU4Rec(config, train_data.dataset).to(config['device'])
    elif config["model"] == 'CLIP_NARM':
        model = CLIP_NARM(config, train_data.dataset).to(config['device'])
    elif config["model"] == 'CLIP_CORE':
        model = CLIP_CORE(config, train_data.dataset).to(config['device'])
    elif config["model"] == 'CLIP_SRGNN':
        model = CLIP_SRGNN(config, train_data.dataset).to(config['device'])
    elif config["model"] == 'CLIP_TAGNN':
        model = CLIP_TAGNN(config, train_data.dataset).to(config['device'])
    elif config["model"] == 'CLIP_GCSAN':
        model = CLIP_GCSAN(config, train_data.dataset).to(config['device'])
    elif config["model"] == 'CLIP_GCEGNN':
        model = CLIP_GCEGNN(config, train_data.dataset).to(config['device'])
    elif config["model"] == 'CLIP_LESSR':
        model = CLIP_LESSR(config, train_data.dataset).to(config['device'])
    elif config["model"] in ['TAGNN','GCEGNN','LESSR']:
        model = get_model_gnn(config["model"])(config, train_data.dataset).to(config["device"])
    else:
        model = get_model(config["model"])(config, train_data.dataset).to(config["device"])

    logger.info(model)

    # trainer loading and initialization
    if config["model"] in ['TAGNN','CLIP_TAGNN','GCEGNN','CLIP_GCEGNN','LESSR','CLIP_LESSR']:
        trainer = get_trainer_gnn(config["MODEL_TYPE"], config["model"])(config, model)
    else:
        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config["show_progress"]
    )

    # model evaluation
    test_result = trainer.evaluate(
        test_data, load_best_model=saved, show_progress=config["show_progress"]
    )

    result = {
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }

    metrics = list(test_result.keys())
    values = list(test_result.values())

    df = pd.DataFrame({metrics[i]:[values[i]] for i in range(len(metrics))})

    if config['n_community']:
        print(f"number of item communities: {config['n_community']}")
        n_community = config['n_community']
    else:
        n_community = 'NA'

    model_n = config['model']
    lr_n = config['learning_rate']

    dataset_n = config['dataset']
    batch_size = config['batch_size']

    output_path = f'./results/results-{model_n}-{dataset_n}.csv'
    if not os.path.exists(output_path): ###
        df.to_csv(output_path, index=False, mode='w')
    else:
        df.to_csv(output_path, index=False, mode='a', header=False)
    print("result saved to "+output_path)
    print(df)
    return result

def run_recs(rank, *args):
    kwargs = args[-1]
    if not isinstance(kwargs, MutableMapping):
        raise ValueError(
            f"The last argument of run_recboles should be a dict, but got {type(kwargs)}"
        )
    kwargs["config_dict"] = kwargs.get("config_dict", {})
    kwargs["config_dict"]["local_rank"] = rank
    run_rec(
        *args[:3],
        **kwargs,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="GRU4Rec",
        help="Model for session-based rec.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="lastfm",
        help="Benchmarks for session-based rec.",
    )
    parser.add_argument('--config_files', type=str, default='seq.yaml', help='config files')
    parser.add_argument("--nproc", type=int, default=1, help="the number of process in this group")
    parser.add_argument("--validation", action="store_true", help="Whether evaluating on validation set (split from train set), otherwise on test set.")
    parser.add_argument("--valid_portion", type=float, default=0.1, help="ratio of validation set.")
    parser.add_argument("--ip", type=str, default="localhost", help="the ip of master node")
    parser.add_argument("--port", type=str, default="5678", help="the port of master node")
    parser.add_argument("--world_size", type=int, default=-1, help="total number of jobs")
    parser.add_argument("--group_offset", type=int, default=0, help="the global rank offset of this group")

    args, _ = parser.parse_known_args()

    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )

    run_session(
        args.model,
        args.dataset,
        config_file_list=config_file_list,
        nproc=args.nproc,
        world_size=args.world_size,
        ip=args.ip,
        port=args.port,
        group_offset=args.group_offset,
    )