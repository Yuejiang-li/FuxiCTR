# =========================================================================
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import sys
import numpy as np
import torch
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import EmbeddingLayer, MLP_Layer, CrossNet
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
import logging

class DCN(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DCN", 
                 gpu=-1, 
                 task="binary_classification",
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 dnn_hidden_units=[], 
                 dnn_activations="ReLU",
                 crossing_layers=3, 
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 **kwargs):
        super(DCN, self).__init__(feature_map, 
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        input_dim = feature_map.num_fields * embedding_dim
        self.dnn = MLP_Layer(input_dim=input_dim,
                             output_dim=None, # output hidden layer
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=None, 
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm, 
                             use_bias=True) \
                   if dnn_hidden_units else None # in case of only crossing net used
        self.crossnet = CrossNet(input_dim, crossing_layers)
        final_dim = input_dim
        if isinstance(dnn_hidden_units, list) and len(dnn_hidden_units) > 0: # if use dnn
            final_dim += dnn_hidden_units[-1]
        self.fc = nn.Linear(final_dim, 1) # [cross_part, dnn_part] -> logit
        self.output_activation = self.get_output_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X, y = self.inputs_to_device(inputs)
        feature_emb = self.embedding_layer(X)
        flat_feature_emb = feature_emb.flatten(start_dim=1)
        cross_out = self.crossnet(flat_feature_emb)
        if self.dnn is not None:
            dnn_out = self.dnn(flat_feature_emb)
            final_out = torch.cat([cross_out, dnn_out], dim=-1)
        else:
            final_out = cross_out
        y_pred = self.fc(final_out)
        if self.output_activation is not None:
            y_pred = self.output_activation(y_pred)
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict


class DCNHard(BaseModel):
    """IncCTR withour KD."""
    def __init__(self, 
                 feature_map, 
                 model_id="DCNHard", 
                 gpu=-1, 
                 task="binary_classification",
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 dnn_hidden_units=[], 
                 dnn_activations="ReLU",
                 crossing_layers=3, 
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 num_experts=3,
                 expert_shape=[],
                 **kwargs):
        super(DCNHard, self).__init__(feature_map, 
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        input_dim = feature_map.num_fields * embedding_dim
        self.dnn = MLP_Layer(input_dim=input_dim,
                             output_dim=None, # output hidden layer
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=None, 
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm, 
                             use_bias=True) \
                   if dnn_hidden_units else None # in case of only crossing net used
        self.crossnet = CrossNet(input_dim, crossing_layers)
        final_dim = input_dim
        if isinstance(dnn_hidden_units, list) and len(dnn_hidden_units) > 0: # if use dnn
            final_dim += dnn_hidden_units[-1]
        self.experts = torch.nn.ModuleList([MLP_Layer(
            input_dim=final_dim,
            output_dim=1,
            hidden_units=expert_shape,
            hidden_activations='ReLU',
            output_activation=None,
            dropout_rates=net_dropout,
            batch_norm=batch_norm,
            use_bias=True
        ) for _ in range(num_experts)])
        self.softmax = torch.nn.Softmax(dim=-1)
        self.output_activation = self.get_output_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def experts_forward(self, X, y):
        feature_emb = self.embedding_layer(X)
        flat_feature_emb = feature_emb.flatten(start_dim=1)
        cross_out = self.crossnet(flat_feature_emb)
        if self.dnn is not None:
            dnn_out = self.dnn(flat_feature_emb)
            final_out = torch.cat([cross_out, dnn_out], dim=-1)
        else:
            final_out = cross_out
        
        # Add experts
        expert_outs = []
        for expert in self.experts:
            if self.output_activation is not None:
                expert_outs.append(self.output_activation(expert(final_out)))
            else:
                expert_outs.append(expert(final_out))

        expert_outs = torch.cat(expert_outs, -1)    # (B, num_experts)

        return expert_outs

    def forward(self, inputs):
        X, y = self.inputs_to_device(inputs)
        expert_outs = self.experts_forward(X, y)    # (B, num_experts)

        y_pred = torch.mean(
            expert_outs,
            dim=-1,
            keepdim=True
        )   # (B, num_experts) -> (B, 1)

        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict


class DCNAdaMoE(DCNHard):
    """AdaMoE is based on hard fusion because it does not work in pretrain stage."""
    def __init__(self, 
                 feature_map, 
                 model_id="DCNAdaMoE", 
                 gpu=-1, 
                 task="binary_classification",
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 dnn_hidden_units=[], 
                 dnn_activations="ReLU",
                 crossing_layers=3, 
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 num_experts=3,
                 expert_shape=[],
                 decay_weights=0.1,
                 **kwargs):

        self.num_experts = num_experts
        self.decay_weights = decay_weights
        expert_weights = torch.ones((1, num_experts), requires_grad=False)
        self.expert_weights = torch.div(expert_weights, torch.sum(expert_weights))
        logging.info("Initial expert_weights: {}".format(self.expert_weights))
        self.R = torch.zeros((num_experts, num_experts), requires_grad=False)
        self.d = torch.zeros((num_experts, 1), requires_grad=False)

        super(DCNAdaMoE, self).__init__(feature_map,
                                        model_id=model_id,
                                        gpu=gpu,
                                        task=task,
                                        learning_rate=learning_rate,
                                        embedding_dim=embedding_dim,
                                        dnn_hidden_units=dnn_hidden_units,
                                        dnn_activations=dnn_activations,
                                        crossing_layers=crossing_layers,
                                        net_dropout=net_dropout,
                                        batch_norm=batch_norm,
                                        embedding_regularizer=embedding_regularizer,
                                        net_regularizer=net_regularizer,
                                        num_experts=num_experts,
                                        expert_shape=expert_shape,
                                        **kwargs)

        self.expert_weights = self.expert_weights.to(self.device)
        self.R = self.R.to(self.device)
        self.d = self.d.to(self.device)

    def forward(self, inputs):
        X, y = self.inputs_to_device(inputs)
        expert_outs = self.experts_forward(X, y)    # (B, num_experts)

        y_pred = torch.sum(
            expert_outs * self.expert_weights,
            dim=-1,
            keepdim=True
        )   # (B, num_experts) -> (B, 1)

        return_dict = {"y_true": y, "y_pred": y_pred, "expert_outs": expert_outs}
        return return_dict

    def evaluate_metrics(self, y_true, y_pred, expert_outs, metrics, **kwargs):
        result = dict()
        for metric in metrics:
            if metric in ['logloss', 'binary_crossentropy']:
                label = y_true[:, None]
                expert_outs[expert_outs < 1e-7] = 1e-7
                expert_outs[expert_outs > (1 - 1e-7)] = 1 - 1e-7
                log_loss = -label * np.log(expert_outs) - (1 - label) * np.log(1 - expert_outs)
                log_loss = np.min(log_loss, axis=-1)
                result[metric] = np.mean(log_loss)
            elif metric == 'AUC':
                result[metric] = roc_auc_score(y_true, y_pred)
            elif metric == "ACC":
                y_pred = np.argmax(y_pred, axis=1)
                result[metric] = accuracy_score(y_true, y_pred)
            else:
                assert "group_index" in kwargs, "group_index is required for GAUC"
                group_index = kwargs["group_index"]
                if metric == "GAUC":
                    pass
                elif metric == "NDCG":
                    pass
                elif metric == "MRR":
                    pass
                elif metric == "HitRate":
                    pass
        logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in result.items()))
        return result

    def evaluate_generator(self, data_generator):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            y_true = []
            expert_outs = []
            if self._verbose > 0:
                from tqdm import tqdm
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)
                y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
                y_true.extend(batch_data[1].data.cpu().numpy().reshape(-1))
                expert_outs.append(return_dict['expert_outs'].clone().detach().cpu().numpy())
            y_pred = np.array(y_pred, np.float64)
            y_true = np.array(y_true, np.float64)
            expert_outs = np.concatenate(expert_outs, axis=0)   # (N, num_experts)
            val_logs = self.evaluate_metrics(y_true, y_pred, expert_outs, self._validation_metrics)
            return val_logs

    def update_experts_weights(self, y_true, expert_outs):
        nt = torch.tensor(y_true.shape[0], device=self.device)
        Y = torch.div(expert_outs, torch.sqrt(nt))    # (B, num_experts)
        y = torch.div(y_true, torch.sqrt(nt))         # (B, 1)
        self.R = self.decay_weights * self.R + torch.matmul(Y.t(), Y)
        self.d = self.decay_weights * self.d + torch.matmul(Y.t(), y)
        new_experts_weights = torch.matmul(torch.inverse(self.R), self.d)
        new_experts_weights[new_experts_weights < 0.1] = 0.1
        new_experts_weights = torch.div(
            new_experts_weights,
            torch.sum(new_experts_weights)
        )
        self.expert_weights = new_experts_weights.t()
        self.expert_weights = self.expert_weights.detach()
        self.R = self.R.detach()
        self.d = self.d.detach()

    def train_one_epoch(self, data_generator, epoch):
        epoch_loss = 0
        self.train()
        batch_iterator = data_generator
        if self._verbose > 0:
            from tqdm import tqdm
            batch_iterator = tqdm(data_generator, disable=False, file=sys.stdout)
        for batch_index, batch_data in enumerate(batch_iterator):
            self.optimizer.zero_grad()
            return_dict = self.forward(batch_data)
            y_true_rep = return_dict['y_true'].expand(-1, self.num_experts)
            loss = torch.functional.F.binary_cross_entropy(return_dict['expert_outs'], y_true_rep, reduction='mean') + \
                self.add_regularization()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
            self.optimizer.step()
            epoch_loss += loss.item()
            self.on_batch_end(batch_index)
            if self._stop_training:
                break
        return epoch_loss / self._batches_per_epoch

    def train_one_epoch_custom(self, data_generator, epoch):
        epoch_loss = 0
        self.train()
        batch_iterator = data_generator
        if self._verbose > 0:
            from tqdm import tqdm
            batch_iterator = tqdm(data_generator, disable=False, file=sys.stdout)
        for batch_index, batch_data in enumerate(batch_iterator):
            self.optimizer.zero_grad()
            return_dict = self.forward(batch_data)
            y_true_rep = return_dict['y_true'].expand(-1, self.num_experts)
            loss = torch.functional.F.binary_cross_entropy(return_dict['expert_outs'], y_true_rep, reduction='mean') + \
                self.add_regularization()
            # loss = self.add_regularization() + \
            #     self.loss_fn(return_dict['y_pred'], return_dict['y_true'], reduction='mean') 
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
            self.optimizer.step()
            epoch_loss += loss.item()
            self.update_experts_weights(return_dict['y_true'], return_dict['expert_outs'])
        return epoch_loss / self._batches_per_epoch
    
    def save_weights(self, checkpoint):
        state_dict = self.state_dict()
        state_dict.update({'expert_weights': self.expert_weights})
        torch.save(state_dict, checkpoint)

    # def load_pretrain(self, checkpoint):
    #     super().load_weights(checkpoint)

    def load_weights(self, checkpoint):
        self.to(self.device)
        state_dict = torch.load(checkpoint, map_location='cpu')
        self.load_state_dict(state_dict, strict=False)

        # load expert_weights
        self.expert_weights.add_((state_dict['expert_weights']).to(self.device) - self.expert_weights)


class DCNAdaMoECE(DCNAdaMoE):
    def __init__(self, 
                 feature_map, 
                 model_id="DCNAdaMoECE", 
                 gpu=-1, 
                 task="binary_classification",
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 dnn_hidden_units=[], 
                 dnn_activations="ReLU",
                 crossing_layers=3, 
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 num_experts=3,
                 expert_shape=[],
                 decay_weights=0.1,
                 **kwargs):

        super(DCNAdaMoECE, self).__init__(feature_map=feature_map,
                                          model_id=model_id,
                                          gpu=gpu,
                                          task=task,
                                          learning_rate=learning_rate,
                                          embedding_dim=embedding_dim,
                                          dnn_hidden_units=dnn_hidden_units,
                                          dnn_activations=dnn_activations,
                                          crossing_layers=crossing_layers,
                                          net_dropout=net_dropout,
                                          batch_norm=batch_norm,
                                          embedding_regularizer=embedding_regularizer,
                                          net_regularizer=net_regularizer,
                                          num_experts=num_experts,
                                          expert_shape=expert_shape,
                                          decay_weights=decay_weights,
                                          **kwargs)
    
    def train_one_epoch_custom(self, data_generator, epoch):
        epoch_loss = 0
        self.train()
        batch_iterator = data_generator
        if self._verbose > 0:
            from tqdm import tqdm
            batch_iterator = tqdm(data_generator, disable=False, file=sys.stdout)
        for batch_index, batch_data in enumerate(batch_iterator):
            self.optimizer.zero_grad()
            return_dict = self.forward(batch_data)
            if not ((return_dict['y_pred'] > 0.0) & (return_dict['y_pred'] <1.0)).all():
                logging.info(f"expert_outs: {return_dict['expert_outs']}")
                logging.info(f"expert_weights: {self.expert_weights}")
                logging.info(f"Anamoly: {return_dict['y_pred']}")
            assert ((return_dict['y_pred'] > 0.0) & (return_dict['y_pred'] <1.0)).all()
            loss = torch.functional.F.binary_cross_entropy(return_dict['y_pred'], return_dict['y_true'], reduction='mean') + \
                self.add_regularization()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
            self.optimizer.step()
            epoch_loss += loss.item()
            self.update_experts_weights(return_dict['y_true'], return_dict['expert_outs'])
        return epoch_loss / self._batches_per_epoch

    def update_experts_weights(self, y_true, expert_outs):
        """
        y_true: shape = (B, 1)
        expert_outs: shape = (B, n_experts)
        """
        y_tilde = y_true * expert_outs + (1 - y_true) * (1 - expert_outs) # (B, n_experts)
        y_tilde = torch.div(y_tilde, torch.sum(y_tilde, dim=-1, keepdim=True) + 1e-8)  # Normalize
        w_cur = torch.mean(y_tilde, dim=0, keepdim=True)    # (1, n_experts)

        self.expert_weights = torch.div(
            self.decay_weights * self.expert_weights + w_cur,
            torch.sum(self.decay_weights * self.expert_weights + w_cur)
        )
        self.expert_weights = self.expert_weights.detach()

class DCNMoE(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DCNMoE", 
                 gpu=-1, 
                 task="binary_classification",
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 dnn_hidden_units=[], 
                 dnn_activations="ReLU",
                 crossing_layers=3, 
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 num_experts=3,
                 expert_shape=[],
                 expert_loss_weights=2.0,
                 **kwargs):
        super(DCNMoE, self).__init__(feature_map, 
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        self.num_experts = num_experts
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        input_dim = feature_map.num_fields * embedding_dim
        self.dnn = MLP_Layer(input_dim=input_dim,
                             output_dim=None, # output hidden layer
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=None, 
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm, 
                             use_bias=True) \
                   if dnn_hidden_units else None # in case of only crossing net used
        self.crossnet = CrossNet(input_dim, crossing_layers)
        final_dim = input_dim
        if isinstance(dnn_hidden_units, list) and len(dnn_hidden_units) > 0: # if use dnn
            final_dim += dnn_hidden_units[-1]
        self.experts = torch.nn.ModuleList([MLP_Layer(
            input_dim=final_dim,
            output_dim=1,
            hidden_units=expert_shape,
            hidden_activations='ReLU',
            output_activation=None,
            dropout_rates=net_dropout,
            batch_norm=batch_norm,
            use_bias=True
        ) for _ in range(num_experts)])
        self.gates = MLP_Layer(
            input_dim=final_dim,
            output_dim=num_experts,
            hidden_units=expert_shape,
            hidden_activations='ReLU',
            output_activation=None,
            dropout_rates=net_dropout,
            batch_norm=batch_norm,
            use_bias=True
        )
        self.softmax = torch.nn.Softmax(dim=-1)
        self.expert_loss_weights = expert_loss_weights
        self.output_activation = self.get_output_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X, y = self.inputs_to_device(inputs)
        feature_emb = self.embedding_layer(X)
        flat_feature_emb = feature_emb.flatten(start_dim=1)
        cross_out = self.crossnet(flat_feature_emb)
        if self.dnn is not None:
            dnn_out = self.dnn(flat_feature_emb)
            final_out = torch.cat([cross_out, dnn_out], dim=-1)
        else:
            final_out = cross_out
        
        # Add experts
        expert_outs = []
        for expert in self.experts:
            if self.output_activation is not None:
                expert_outs.append(self.output_activation(expert(final_out)))
            else:
                expert_outs.append(expert(final_out))

        expert_outs = torch.cat(expert_outs, -1)    # (B, num_experts)
        expert_weights = self.softmax(self.gates(final_out)) # (B, num_experts)
        self.expert_weights = torch.mean(expert_weights, dim=0)    # (num_experts)
        y_pred = torch.sum(
            expert_outs * expert_weights,
            dim=-1,
            keepdim=True
        ).clamp(0.0, 1.0)   # (B, num_experts) -> (B, 1)

        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict

    def get_total_loss(self, inputs):
        normal_loss = super().get_total_loss(inputs=inputs)
        if self.num_experts == 1:
            total_loss = normal_loss
        else:
            expert_loss = torch.std(self.expert_weights)
            total_loss = normal_loss + self.expert_loss_weights * expert_loss

        return total_loss


class DCNIADM(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DCN", 
                 gpu=-1, 
                 task="binary_classification",
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 dnn_hidden_units=[], 
                 dnn_activations="ReLU",
                 crossing_layers=3, 
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 adapt_dim=64, 
                 num_experts=3,
                 expert_shape=[],
                 expert_loss_weights=2.0,
                 **kwargs):
        super(DCNIADM, self).__init__(feature_map, 
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        self.num_experts = num_experts
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        input_dim = feature_map.num_fields * embedding_dim
        self.dnn = MLP_Layer(input_dim=input_dim,
                             output_dim=None, # output hidden layer
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=None, 
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm, 
                             use_bias=True) \
                   if dnn_hidden_units else None # in case of only crossing net used
        self.crossnet = CrossNet(input_dim, crossing_layers)
        final_dim = input_dim
        if isinstance(dnn_hidden_units, list) and len(dnn_hidden_units) > 0: # if use dnn
            final_dim += dnn_hidden_units[-1]
        self.adapt_layer = torch.nn.Linear(final_dim, adapt_dim)
        self.iadm_layers = torch.nn.ModuleList([
            torch.nn.Linear(adapt_dim, adapt_dim)
            for _ in range(num_experts)
        ])
        self.experts = torch.nn.ModuleList([MLP_Layer(
            input_dim=adapt_dim,
            output_dim=1,
            hidden_units=expert_shape,
            hidden_activations='ReLU',
            output_activation=None,
            dropout_rates=net_dropout,
            batch_norm=batch_norm,
            use_bias=True
        ) for _ in range(num_experts)])
        self.gates = MLP_Layer(
            input_dim=adapt_dim,
            output_dim=num_experts,
            hidden_units=expert_shape,
            hidden_activations='ReLU',
            output_activation=None,
            dropout_rates=net_dropout,
            batch_norm=batch_norm,
            use_bias=True
        )
        self.softmax = torch.nn.Softmax(dim=-1)
        self.relu = torch.nn.ReLU()
        self.expert_loss_weights = expert_loss_weights
        self.output_activation = self.get_output_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X, y = self.inputs_to_device(inputs)
        feature_emb = self.embedding_layer(X)
        flat_feature_emb = feature_emb.flatten(start_dim=1)
        cross_out = self.crossnet(flat_feature_emb)
        if self.dnn is not None:
            dnn_out = self.dnn(flat_feature_emb)
            final_out = torch.cat([cross_out, dnn_out], dim=-1)
        else:
            final_out = cross_out
        
        final_out = self.adapt_layer(final_out) # (B, adapt_size)
        expert_weights = self.softmax(self.gates(final_out)) # (B, num_experts)
        self.expert_weights = torch.mean(expert_weights, dim=0)    # (num_experts)
        # Add experts
        expert_outs = []
        for iadm_layer, expert in zip(self.iadm_layers, self.experts):
            final_out = self.relu(iadm_layer(final_out))    # (B, adapt_size)
            if self.output_activation is not None:
                expert_outs.append(self.output_activation(expert(final_out)))
            else:
                expert_outs.append(expert(final_out))            

        expert_outs = torch.cat(expert_outs, -1)    # (B, num_experts)
        y_pred = torch.sum(
            expert_outs * expert_outs,
            dim=-1,
            keepdim=True
        ).clamp(0.0, 1.0)   # (B, num_experts) -> (B, 1)

        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict

    def get_total_loss(self, inputs):
        normal_loss = super().get_total_loss(inputs=inputs)
        if self.num_experts == 1:
            total_loss = normal_loss
        else:
            expert_loss = torch.std(self.expert_weights)
            total_loss = normal_loss + self.expert_loss_weights * expert_loss

        return total_loss

