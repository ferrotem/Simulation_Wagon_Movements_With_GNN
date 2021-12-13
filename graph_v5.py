
#%%
import torch
from torch._C import _enable_minidumps
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import Conv2d
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch_geometric.data import Data

import torch
from torch_geometric.data import Data

import pandas as pd

edge_index = torch.tensor([[3,2,4, 1, 1],
                           [1,1,3, 4, 4]], dtype=torch.long)
edge_attribute =torch.tensor([[[5,1],[1,0],[2,0],[5,1],[1,0]],  # a->b [n_wags, delay]
                              [[3,2],[4,2],[0,0],[7,1],[5,0]]]) 
# x = torch.tensor([[0],[1], [1], [3], [1]], dtype=torch.float)
x = torch.tensor([[0,0],[1,0], [1,0], [3,0], [1,0]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index, edge_attr=edge_attribute)

orders  = pd.DataFrame(edge_index.numpy().T, columns=['source', 'dest'])
orders = orders.drop_duplicates()
orders['price'] = [500,100,100,800]
orders['por_cost'] = [100,200,200,200]
orders['gruzh_cost'] = [5,10,15,20]
def get_price(source_st, dest_st):
    try:
        k = orders[(orders.source ==source_st)&(orders.dest==dest_st)]
        return k['price'] - k['por_cost'] - k['gruzh_cost']
    except:
        return 0
income_list = []
for i in range(edge_index.shape[1]):
    income_list.append(get_price(edge_index[0][i].numpy(),edge_index[1][i].numpy()).values[0])
income_list = torch.tensor(income_list)

# %%
## GNN Tutorial 
import torch
from torch_geometric.nn import  MessagePassing
import math

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0/(tensor.size(-2)+tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def add_self_loops(edge_index, num_nodes=None):
    loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device) # [0, N]
    loop_index = loop_index.unsqueeze(0).repeat(2,1) # [[0, 0],
                                                      # [N, N]]
    edge_index = torch.cat([edge_index, loop_index], dim=1) 
    
    return edge_index

def degree(index, num_nodes=None, dtype=None):
    out = torch.zeros((num_nodes), dtype=dtype, device=index.device)
    return out.scatter_add_(0, index, out.new_ones((index.size(0))))

def balance_subract(index, edge_attr, num_nodes=None, dtype=None):

    edge_attr = torch.sum(edge_attr[:,:,0], 0)

    print("subructor", edge_attr)
    out = torch.zeros((num_nodes), dtype=dtype, device=index.device)
    if index[0]==0:
        return out
    else:
        return out.scatter_add_(0, index, edge_attr)

def filter_to_add_balance(edge_attr):
    filtered_edge = edge_attr.clone().detach() 
    filter_= filtered_edge[:,:,1]==0
    filter_ = filter_.int()        
    filtered_edge[:,:, 0] = filtered_edge[:,:, 0]*filter_

    return filtered_edge

def filter_out_expired_edge_attr(edge_attr):
    filter_= edge_attr[:,:,1]!=-1
    filter_ = filter_.to(torch.long)
    edge_attr[:,:,0] = torch.mul(filter_, edge_attr[:,:,0])
    return edge_attr
def update_edge_attr(edge_attr):
    edge_attr[:,:,1]-=1
    return edge_attr # Implementing the GCN
# Steps:
    # 1. Add self-loops to the adjacency matrix
    # 2. Linearly transform node feature matrix
    # 3. Normalize node features
    # 4. Sum up neighboring node features
    # 5. Return new node embedding
#%%

class GCNconv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.reset_parameters()
    
    def reset_parameters(self):
        glorot(self.lin.weight)
        zeros(self.lin.bias)
    
    def forward(self, x, edge_index, edge_attr):
        # x has shape [N, in_channels] N- number of nodes
        # edge_index has shape [2,E]

        # Step 1
        # edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2 
        # x = self.lin(x)

        # Step 3-5 Start propagating messages
        return self.propagate(edge_index, edge_attr= edge_attr, x=x)


    def message(self,x_j, x_i, edge_attr, edge_index, size):
        # x_j has shape [E, out_channels]
        new_ed_attr = filter_to_add_balance(edge_attr)
        income = income_list*new_ed_attr[0,:,0]
        income = torch.unsqueeze(income, 1) 

        new_ed_attr = torch.sum(new_ed_attr[:,:, 0], 0)
        new_ed_attr = new_ed_attr.view(-1,1)
        
        return torch.cat([new_ed_attr, income], axis=1)

    
    def update(self, aggr_out, edge_index, edge_attr, x):
        # aggr_out has shape [N, out_channels]
        # Step 5: Return new node embeddings
        row, col = edge_index
        
        # deg_attr = filter_to_add_balance(edge_attr)
        deg = balance_subract(row, edge_attr=edge_attr, num_nodes=len(x), dtype=aggr_out.dtype)
        edge_index[0,:] =0
        edge_attr = update_edge_attr(edge_attr)

        deg = deg.view(-1,1)
        x[:,0] = x[:,0] + aggr_out[:, 0]-deg[:,0]
        x[:,1] = x[:,1] + aggr_out[:, 1]
        aggr_out = x

        print("edge_attr: ", edge_attr)
        return aggr_out, edge_attr
#%%
conv = GCNconv(1,1)
x, edge_attr=conv(data.x, data.edge_index, data.edge_attr)
edge_attr = filter_out_expired_edge_attr(edge_attr)
print("HI", x, edge_attr)

x, edge_attr=conv(x, data.edge_index, edge_attr)
edge_attr = filter_out_expired_edge_attr(edge_attr)
print("HI 2 ",x, edge_attr)

x, edge_attr=conv(x, data.edge_index, edge_attr)
edge_attr = filter_out_expired_edge_attr(edge_attr)
print("HI 3 ",x, edge_attr)
 



























# %%
