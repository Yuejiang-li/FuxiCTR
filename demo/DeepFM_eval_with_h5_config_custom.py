import sys
sys.path.append('../')
import os
import logging
from datetime import datetime
from fuxictr import datasets
from fuxictr.features import FeatureMap
from fuxictr.utils import load_config, set_logger, print_to_json
from fuxictr.pytorch.models import DeepFM
from fuxictr.pytorch.torch_utils import seed_everything


if __name__ == '__main__':
    # Load params from config files
    config_dir = 'taobao_all_config'
    experiment_id = 'DeepFM_seq_test_h5_custom' # correponds to h5 input `taobao_tiny_h5`
    params = load_config(config_dir, experiment_id)

    # set up logger and random seed
    set_logger(params, eval=True)
    logging.info(print_to_json(params))
    seed_everything(seed=params['seed'])

    # Load feature_map from json
    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(os.path.join(data_dir, "feature_map.json"))

    # Get train and validation data generator from h5
    test_gen = datasets.h5_generator(feature_map,
                                     stage='test',
                                     test_data=os.path.join(data_dir, 'test.h5'),
                                     batch_size=2048,
                                     shuffle=False
    )
    
    # Model initialization and fitting                                                  
    model = DeepFM(feature_map, **params)
    # Reloading weights of the best checkpoint
    logging.info("Reload model parameter from {}".format(model.checkpoint))
    model.load_weights(model.checkpoint)
    model.count_parameters() # print number of parameters used in model
    logging.info('***** test results *****')
    model.evaluate_generator(test_gen)
