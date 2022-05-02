import sys
sys.path.append('../')
import argparse
import os
import logging
from datetime import datetime
from fuxictr import datasets
from fuxictr.features import FeatureMap
from fuxictr.utils import load_config, set_logger, print_to_json
from fuxictr.pytorch import models
from fuxictr.pytorch.torch_utils import seed_everything

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expid', type=str, default='FM_test', help='The experiment id to run.')
    parser.add_argument('--dataset', type=str, default='avazu', choices=['avazu', 'taobao'])
    
    args = vars(parser.parse_args())

    return args


if __name__ == '__main__':
    args = get_args()
    experiment_id = args['expid']
    if args['dataset'] == 'taobao':
        config_dir = 'taobao_all_config'
    else:
        config_dir = 'avazu_all_config'

    params = load_config(config_dir, experiment_id)

    # set up logger and random seed
    set_logger(params)
    logging.info(print_to_json(params))
    seed_everything(seed=params['seed'])

    logging.info("Command args: {}".format(args))

    # Load feature_map from json
    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(os.path.join(data_dir, "feature_map.json"))

    # Get train and validation data generator from h5
    train_gen, valid_gen = datasets.h5_generator(feature_map, 
                                                 stage='train', 
                                                 train_data=params['train_data'],
                                                 valid_data=params['valid_data'],
                                                 batch_size=params['batch_size'],
                                                 shuffle=params['shuffle'])
    
    # Model initialization and fitting                                                  
    model_class = getattr(models, params['model'])
    model = model_class(feature_map, **params)
    model.count_parameters() # print number of parameters used in model
    model.fit_generator(train_gen, 
                        validation_data=valid_gen, 
                        epochs=params['epochs'],
                        verbose=params['verbose'])
    model.load_weights(model.checkpoint) # reload the best checkpoint
    
    logging.info('***** validation results *****')
    model.evaluate_generator(valid_gen)
