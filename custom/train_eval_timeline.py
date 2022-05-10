import sys

sys.path.append('../')
import argparse
import os
import logging
from datetime import datetime
import pandas as pd
from fuxictr import datasets
from fuxictr.features import FeatureMap
from fuxictr.utils import load_config, set_logger, print_to_json
from fuxictr.pytorch import models
from fuxictr.pytorch.torch_utils import seed_everything

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expid', type=str, default='FM_test', help='The experiment id to run.')
    parser.add_argument('--dataset', type=str, default='avazu', choices=['avazu', 'taobao'])
    parser.add_argument('--gpu', type=int, default=-1)
    
    args = vars(parser.parse_args())

    return args

def train_one_dataset(model, data_dir, dataset,
    batch_size, epochs, verbose, shuffle):
    train_gen = datasets.h5_train_generator(
        train_data=os.path.join(data_dir, dataset),
        batch_size=batch_size,
        shuffle=shuffle,
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
    args = get_args()
    experiment_id = args['expid']
    if args['dataset'] == 'taobao':
        config_dir = 'taobao_all_config'
    else:
        config_dir = 'avazu_all_config'
    params = load_config(config_dir, experiment_id)
    params['gpu'] = args['gpu']

    # set up logger and random seed
    set_logger(params)
    logging.info(print_to_json(params))
    seed_everything(seed=params['seed'])
    logging.info("Command args: {}".format(args))

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
    model_class = getattr(models, params['model'])
    model = model_class(feature_map, **params)
    pretrain_model = params.get("pretrain_model", None)
    if pretrain_model is not None:
        logging.info("Load pretrained model from {}".format(pretrain_model))
        if hasattr(model, 'load_pretrain'):
            logging.info('Find implemented load_pretrain.')
            model.load_pretrain(pretrain_model)
        else:
            logging.info('Not find implemented load_pretrain.')
            model.load_weights(pretrain_model)

    # Loop over all timeline datasets chronologically
    start_time = datetime.now()
    eval_timeline = {x: [] for x in params['metrics']}
    for i, timeline_dataset in enumerate(timeline_datasets):
        if i == 0:
            # The initial dataset dose not need validation.
            train_one_dataset(model, data_dir, timeline_dataset,
                params["batch_size"], params['epochs'], params['verbose'], params['shuffle'])
        else:
            # First use new data to evaluate.
            eval_res = eval_one_dataset(model, feature_map, data_dir, timeline_dataset,
                params['batch_size'])
            for k, v in eval_res.items():
                eval_timeline[k].append(v)

            # Next train with new data
            train_one_dataset(model, data_dir, timeline_dataset,
                params["batch_size"], params['epochs'], params['verbose'], params['shuffle'])

    # Save model.
    model.save_weights(model.checkpoint)

    # Save evaluation results.
    cur_time = datetime.now().strftime("%Y%m%d%H%M%S")
    if 'eval_res_name' not in params:
        eval_res_path = '../eval'
        if not os.path.exists(eval_res_path):
            os.mkdir(eval_res_path)
        eval_res_name = os.path.join(eval_res_path, f"{cur_time}_{experiment_id}.csv")
    else:
        eval_res_name = params['eval_res_name']
    
    logging.info("Total time consumes: {:.2f}"
        .format((datetime.now() - start_time).total_seconds()))

    eval_df = pd.DataFrame(eval_timeline)
    eval_df.to_csv(eval_res_name, index=False)
