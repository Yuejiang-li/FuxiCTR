import os
import sys
sys.path.append('../')
import os
from fuxictr import datasets
from fuxictr.datasets.taobao import FeatureEncoder
from fuxictr.utils import set_logger, print_to_json, load_dataset_config
import logging

if __name__ == '__main__':
    # Load params from config files
    config_dir = 'taobao_all_config'
    dataset_id = 'taobao_timeline'
    params = load_dataset_config(config_dir, dataset_id)

    # set up logger and random seed
    set_logger(params, log_file='./demo.log')
    logging.info(print_to_json(params))

    # Set up feature encoder
    feature_encoder = FeatureEncoder(feature_cols=params["feature_cols"],
                                    label_col=params["label_col"],
                                    dataset_id=dataset_id, 
                                    data_root=params["data_root"])
    datasets.build_timeline_dataset(feature_encoder, 
                        train_data=params["train_data"])
