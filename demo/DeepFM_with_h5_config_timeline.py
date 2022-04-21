import sys

sys.path.append('../')
import os
import logging
from datetime import datetime
import pandas as pd
from fuxictr import datasets
from fuxictr.features import FeatureMap
from fuxictr.utils import load_config, set_logger, print_to_json
from fuxictr.pytorch.models import DeepFM
from fuxictr.pytorch.torch_utils import seed_everything


def train_one_dataset(model, data_dir, dataset,
    batch_size, epochs, verbose):
    train_gen = datasets.h5_train_generator(
        train_data=os.path.join(data_dir, dataset),
        batch_size=batch_size,
        shuffle=False,
    )
    model.fit_generator_stream(train_gen, epochs=epochs, verbose=verbose)


def eval_one_dataset(model, feature_map, data_dir, dataset,
    batch_size):
    test_gen = datasets.h5_generator(
        feature_map,
        test_data=os.path.join(data_dir, dataset),
        batch_size=batch_size,
        shuffle=False,
        stage='test'
    )
    eval_res = model.evaluate_generator(test_gen)

    return eval_res


if __name__ == '__main__':
    # Load params from config files
    config_dir = 'taobao_all_config'
    experiment_id = 'DeepFM_timeline_h5' # correponds to h5 input `taobao_tiny_h5`
    params = load_config(config_dir, experiment_id)

    # set up logger and random seed
    set_logger(params)
    logging.info(print_to_json(params))
    seed_everything(seed=params['seed'])

    # Load feature_map from json
    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(os.path.join(data_dir, "feature_map.json"))

    # List all .h5 timeline files and sort
    timeline_datasets = os.listdir(data_dir)
    timeline_datasets = [x for x in timeline_datasets if x[-2:] == 'h5']
    timeline_datasets.sort()
    logging.info("Number of timeline datasets in .h5 format: {}".format(len(timeline_datasets)))

    # Build Model
    model = DeepFM(feature_map, **params)

    # Loop over all timeline datasets chronologically
    start_time = datetime.now()
    eval_timeline = {x: [] for x in params['metrics']}
    for i, timeline_dataset in enumerate(timeline_datasets):
        if i == 0:
            # The initial dataset dose not need validation.
            train_one_dataset(model, data_dir, timeline_dataset,
                params["batch_size"], 1, params['verbose'])
        else:
            # First use new data to evaluate.
            eval_res = eval_one_dataset(model, feature_map, data_dir, timeline_dataset,
                params['batch_size'])
            for k, v in eval_res.items():
                eval_timeline[k].append(v)

            # Next train with new data
            train_one_dataset(model, data_dir, timeline_dataset,
                params["batch_size"], 1, params['verbose'])

    # Save model.
    model.save_weights(model.checkpoint)

    # Save evaluation results.
    eval_res_path = '../eval'
    if not os.path.exists(eval_res_path):
        os.mkdir(eval_res_path)
    cur_time = datetime.now().strftime("%Y%m%d%H%M%S")
    logging.info("Total time consumes: {:.2f}"
        .format((datetime.now() - start_time).total_seconds()))
    eval_res_name = os.path.join(eval_res_path, f"{cur_time}_{experiment_id}.csv")
    eval_df = pd.DataFrame(eval_timeline)
    eval_df.to_csv(eval_res_name, index=False)
