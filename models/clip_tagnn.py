# @Time   : 2022/3/17
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

r"""
TAGNN
################################################

Reference:
    Feng Yu et al. "TAGNN: Target Attentive Graph Neural Networks for Session-based Recommendation." in SIGIR 2020 short.
    Implemented using PyTorch Geometric.

Reference code:
    https://github.com/CRIPAC-DIG/TAGNN

"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender

from recbole_gnn.model.layers import SRGNNCell
import pickle

class CLIP_TAGNN(SequentialRecommender):
    r"""TAGNN introduces target-aware attention and adaptively activates different user interests with respect to varied target items.
    """

    def __init__(self, config, dataset):
        super(CLIP_TAGNN, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.step = config['step']
        self.device = config['device']
        self.loss_type = config['loss_type']

        # item embedding
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)

        ###
        self.data_path = config['data_path']
        with open(f'{self.data_path}/partition.pickle', 'rb') as f:
            self.partition = pickle.load(f)
        
        partition_values = list(self.partition.values())
        self.most_frequent_partition = max(set(partition_values), key=partition_values.count)

        self.n_communities = len(set(self.partition.values()))
        self.community_prompt = nn.Embedding(self.n_communities + 1, self.embedding_size, padding_idx=0)

        # gate layers
        self.gate_layer_item = nn.Linear(self.embedding_size, self.embedding_size)
        self.gate_layer_prompt = nn.Linear(self.embedding_size, self.embedding_size)
        self.sigmoid = nn.Sigmoid()
        ###

        # define layers and loss
        self.gnncell = SRGNNCell(self.embedding_size)
        self.linear_one = nn.Linear(self.embedding_size, self.embedding_size)
        self.linear_two = nn.Linear(self.embedding_size, self.embedding_size)
        self.linear_three = nn.Linear(self.embedding_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.embedding_size * 2, self.embedding_size)
        self.linear_t = nn.Linear(self.embedding_size, self.embedding_size, bias=False)  #target attention
        if self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, edge_index, alias_inputs, item_seq_len, item_seq):
        mask = alias_inputs.gt(0)
        hidden = self.item_embedding(x)

        for i in range(self.step):
            hidden = self.gnncell(hidden, edge_index)

        seq_hidden = hidden[alias_inputs]

        ###
        # Step 1: Flatten the item_seq for easier processing
        item_seq_flat = item_seq.flatten()

        # Step 2: Convert item_id to community_id using self.partition
        community_seq_flat = torch.zeros_like(item_seq_flat, device=item_seq.device)  # Initialize with zeros
        non_zero_mask = item_seq_flat != 0  # Identify non-zero elements
        
        # Only convert non-zero elements
        community_ids = [self.partition.get(item_id.item(), 0) for item_id in item_seq_flat[non_zero_mask]]
        community_seq_flat[non_zero_mask] = torch.tensor(community_ids, device=item_seq.device)

        # Step 3: Reshape back to the original shape
        community_seq = community_seq_flat.view_as(item_seq)

        seq_len = alias_inputs.size(1)  # Get the second dimension size of alias_inputs
        community_id_selected = community_seq[:, :seq_len] 
        cprompt = self.community_prompt(community_id_selected)

        item_normalized = F.normalize(seq_hidden, p=2, dim=1) 
        cprompt_normalized = F.normalize(cprompt, p=2, dim=1)
        # item_normalized = seq_hidden
        # cprompt_normalized = cprompt

        # Compute gating weights
        gate_weight_item = self.sigmoid(self.gate_layer_item(item_normalized))
        gate_weight_cprompt = self.sigmoid(self.gate_layer_prompt(cprompt_normalized))

        seq_hidden = gate_weight_item * item_normalized + gate_weight_cprompt * cprompt_normalized

        # fetch the last hidden state of last timestamp
        ht = self.gather_indexes(seq_hidden, item_seq_len - 1)
        q1 = self.linear_one(ht).view(ht.size(0), 1, ht.size(1))
        q2 = self.linear_two(seq_hidden)

        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        alpha = F.softmax(alpha, 1)
        a = torch.sum(alpha * seq_hidden * mask.view(mask.size(0), -1, 1).float(), 1)
        seq_output = self.linear_transform(torch.cat([a, ht], dim=1))

        seq_hidden = seq_hidden * mask.view(mask.shape[0], -1, 1).float()
        qt = self.linear_t(seq_hidden)
        b = self.item_embedding.weight
        beta = F.softmax(b @ qt.transpose(1,2), -1)
        target = beta @ seq_hidden
        a = seq_output.view(ht.shape[0], 1, ht.shape[1])  # b,1,d
        a = a + target  # b,n,d
        scores = torch.sum(a * b, -1)  # b,n
        return scores

    def calculate_loss(self, interaction):
        x = interaction['x']
        edge_index = interaction['edge_index']
        alias_inputs = interaction['alias_inputs']
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        logits = self.forward(x, edge_index, alias_inputs, item_seq_len, interaction['item_id_list'])
        pos_items = interaction[self.POS_ITEM_ID]
        loss = self.loss_fct(logits, pos_items)
        return loss

    def predict(self, interaction):
        pass

    def full_sort_predict(self, interaction):
        x = interaction['x']
        edge_index = interaction['edge_index']
        alias_inputs = interaction['alias_inputs']
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        scores = self.forward(x, edge_index, alias_inputs, item_seq_len, interaction['item_id_list'])
        return scores
