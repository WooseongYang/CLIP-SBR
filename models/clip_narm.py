# -*- coding: utf-8 -*-
# @Time   : 2020/8/25 19:56
# @Author : Yujie Lu
# @Email  : yujielu1998@gmail.com

# UPDATE
# @Time   : 2020/9/15, 2020/10/2
# @Author : Yupeng Hou, Yujie Lu
# @Email  : houyupeng@ruc.edu.cn, yujielu1998@gmail.com

r"""
NARM
################################################

Reference:
    Jing Li et al. "Neural Attentive Session-based Recommendation." in CIKM 2017.

Reference code:
    https://github.com/Wang-Shuo/Neural-Attentive-Session-Based-Recommendation-PyTorch

"""

import torch
from torch import nn
from torch.nn.init import xavier_normal_, constant_
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss

import pickle
import numpy as np
import pandas as pd

class CLIP_NARM(SequentialRecommender):
    r"""NARM explores a hybrid encoder with an attention mechanism to model the user’s sequential behavior,
    and capture the user’s main purpose in the current session.

    """

    def __init__(self, config, dataset):
        super(CLIP_NARM, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.hidden_size = config["hidden_size"]
        self.n_layers = config["n_layers"]
        self.dropout_probs = config["dropout_probs"]
        self.device = config["device"]

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.embedding_size, padding_idx=0
        )

        ###
        self.data_path = config['data_path']
        with open(f'{self.data_path}/partition.pickle', 'rb') as f:
            self.partition = pickle.load(f)
        
        partition_values = list(self.partition.values())
        self.most_frequent_partition = max(set(partition_values), key=partition_values.count)

        # Create a lookup tensor with default values set to the most frequent partition
        max_item_id = max(self.partition.keys())
        self.lookup_tensor = torch.full((max_item_id + 1,), self.most_frequent_partition, device=self.device, dtype=torch.long)

        # Fill the lookup tensor with the partition values
        for item_id, community_id in self.partition.items():
            self.lookup_tensor[item_id] = community_id
        
        # self.n_communities = max(dataset.inter_feat['community_id']) 
        self.n_communities = len(set(self.partition.values()))
        self.community_prompt = nn.Embedding(self.n_communities + 1, self.embedding_size, padding_idx=0)

        # gate layers
        self.gate_layer_item = nn.Linear(self.embedding_size, self.embedding_size)
        self.gate_layer_prompt = nn.Linear(self.embedding_size, self.embedding_size)
        self.sigmoid = nn.Sigmoid()
        ###

        self.emb_dropout = nn.Dropout(self.dropout_probs[0])
        self.gru = nn.GRU(
            self.embedding_size,
            self.hidden_size,
            self.n_layers,
            bias=False,
            batch_first=True,
        )
        self.a_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_t = nn.Linear(self.hidden_size, 1, bias=False)
        self.ct_dropout = nn.Dropout(self.dropout_probs[1])
        self.b = nn.Linear(2 * self.hidden_size, self.embedding_size, bias=False)
        self.loss_type = config["loss_type"]
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, item_seq, item_seq_len):
        item_seq_emb = self.item_embedding(item_seq)

        # Flatten the item_seq for easier processing
        item_seq_flat = item_seq.flatten()

        # Create a mask for zero and out-of-bound elements
        zero_mask = item_seq_flat == 0
        out_of_bound_mask = item_seq_flat >= self.lookup_tensor.size(0)

        # Apply the lookup tensor and handle out-of-bound elements
        community_seq_flat = self.lookup_tensor[item_seq_flat.clamp(max=max(self.partition.keys()))]

        # Replace zeros and out-of-bound elements with the most frequent partition
        community_seq_flat = torch.where(zero_mask, torch.tensor(0, device=item_seq.device), community_seq_flat)

        # Reshape back to original shape
        community_seq = community_seq_flat.view_as(item_seq)

        cprompt = self.community_prompt(community_seq)

        item_normalized = F.normalize(item_seq_emb, p=2, dim=1) 
        cprompt_normalized = F.normalize(cprompt, p=2, dim=1)
        # item_normalized = item_seq_emb
        # cprompt_normalized = cprompt

        # Compute gating weights
        gate_weight_item = self.sigmoid(self.gate_layer_item(item_normalized))
        gate_weight_cprompt = self.sigmoid(self.gate_layer_prompt(cprompt_normalized))

        gated_input = gate_weight_item * item_normalized + gate_weight_cprompt * cprompt_normalized

        item_seq_emb_dropout = self.emb_dropout(gated_input)
        gru_out, _ = self.gru(gated_input)

        # fetch the last hidden state of last timestamp
        c_global = ht = self.gather_indexes(gru_out, item_seq_len - 1)
        # avoid the influence of padding
        mask = item_seq.gt(0).unsqueeze(2).expand_as(gru_out)
        q1 = self.a_1(gru_out)
        q2 = self.a_2(ht)
        q2_expand = q2.unsqueeze(1).expand_as(q1)
        # calculate weighted factors α
        alpha = self.v_t(mask * torch.sigmoid(q1 + q2_expand))
        c_local = torch.sum(alpha.expand_as(gru_out) * gru_out, 1)
        c_t = torch.cat([c_local, c_global], 1)
        c_t = self.ct_dropout(c_t)
        seq_output = self.b(c_t)
        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores
