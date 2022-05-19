# AdaMoE

Our code is in `./custom`.

To run the code, please check the following steps:

1. Download Avazu data as illustrated in https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/datasets/Avazu

2. `cd custom` and run `python preprocess_h5_avazu_timeline.py` to obtain the dataset in stream form

3. Pretain model. The pretrain configuration yaml files are in `./custom/avazu_all_config/model_config/*pretrain.yaml`. `cd custom` and run `python timeline_pretrain.py --expid <expid>` to obtain the certain pretrained model.

4. Stream training. The stream training yaml files are in `./custom/avazu_all_config/model_config/*timeline.yaml`. `cd custom` and run `python train_eval_timeline.py --expid <expid>` to obtain the results of a certain exp. The evaluation results is in `./eval` in csv format, with each record being the AUC and logloss of the corresponding stream dataset.

Note that the `data_path`, `pretrain_model`, etc., configuration should be adjust acccordingly.
