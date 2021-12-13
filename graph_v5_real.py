
#%%
from collections import deque
import torch
from torch._C import _enable_minidumps, dtype
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import Conv2d
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch_geometric.data import Data

import torch
from torch_geometric.data import Data

import pandas as pd
import numpy as np
#%%
df = pd.read_pickle("./data/vagon_transfers_for_graph.pkl")

def gen_stations_map(df):
    stations_list = list(set(df.from_station.unique().tolist()+df.dest_station.unique().tolist()))
    stations_list.sort()
    assert len(stations_list)==649
    stations_map = {'Empty':0}
    for i in range(len(stations_list)):
        stations_map[stations_list[i]]=i+1
    node_list = list(stations_map.values())
    assert len(node_list)==len(stations_list)+1
    return node_list, stations_map

node_list, stations_map  = gen_stations_map(df)

def preprocess_dataset(df):
    df['gruzh']= df[df.grzh_por=='ГРУЖ']['vagon']
    df['por']= df[df.grzh_por=='ПОР']['vagon']
    df = df.groupby(['from_station' ,'dest_station',"mdate"]).sum().reset_index()
    df =df.sort_values(by=['mdate'])
    df.loc[:, 'code_a'] = df.loc[:,'from_station'].apply(lambda x: stations_map[x])
    df.loc[:, 'code_b'] = df.loc[:,'dest_station'].apply(lambda x: stations_map[x])
    date_list = df.mdate.unique().tolist()
    return df, date_list


def calc_edges(df, mdate):
    rf = df[df.mdate==mdate].copy()

    rf[['por','gruzh']]=rf[['por','gruzh']].astype('int')
    rf = rf.fillna(0)
    rf['gruzh_dur']=3
    rf['por_dur']=1

    edge_index = torch.tensor(rf[['code_a', 'code_b']].values.T, dtype=torch.long)
    edge_attribute = torch.tensor([rf[['gruzh','gruzh_dur']].values, rf[['por','por_dur']].values], dtype=torch.long)
    return edge_index, edge_attribute
# uncomment to get one day result
df, date_list = preprocess_dataset(df)
# df = df[(df.from_station.isin(['Достык','Алматы 1']))&(df.dest_station.isin(['Достык','Алматы 1']))]

# edge_index, edge_attribute = calc_edges(df, date_list[0])

# def add_initial_balance(node_list):
#     return [[idx, 0] for idx in node_list]

# x = add_initial_balance(node_list=node_list)
# x = torch.tensor(x, dtype=torch.float)

#%%

total_edge_index = df[['code_a', 'code_b']].drop_duplicates().values

orders  = pd.DataFrame(total_edge_index, columns=['source', 'dest'])
total_edge_index = total_edge_index.T
print(total_edge_index.shape)
orders = orders.drop_duplicates()
orders['price'] = np.zeros(total_edge_index.shape[1], dtype=np.int16)#[500,100,100,800]
orders['por_cost'] = np.zeros(total_edge_index.shape[1], dtype=np.int16)#[100,200,200,200]
orders['gruzh_cost'] = np.zeros(total_edge_index.shape[1], dtype=np.int16)#[5,10,15,20]
#%%
def get_price(source_st, dest_st):
    k = orders[(orders.source ==source_st)&(orders.dest==dest_st)]
    if len(k)>0:
        income =  k['price'] - k['por_cost'] - k['gruzh_cost']
        # print(income.values[0])
        return income.values[0]
    else:
        return 0
def calc_income_list(edge_index):
    income_list = []
    for i in range(edge_index.shape[1]):
        income = get_price(edge_index[0][i].numpy(),edge_index[1][i].numpy())
        income_list.append(income)
    income_list = torch.tensor(income_list)
    return income_list

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

    # print("subructor", edge_attr)
    out = torch.zeros((num_nodes), dtype=dtype, device=index.device)
    # if index[0]==0:
    #     return out
    # else:
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
        # val_ed = deg[166]
        edge_index[0,:] =0
        edge_attr = update_edge_attr(edge_attr)

        deg = deg.view(-1,1)
        # val_nod = aggr_out[29, 0]
        x[:,0] = x[:,0] + aggr_out[:, 0]-deg[:,0]
        x[:,1] = x[:,1] + aggr_out[:, 1]
        aggr_out = x

        #print("edge_attr: ", edge_attr)
        return aggr_out, edge_attr
#%%
# conv = GCNconv(1,1)
# x, edge_attr=conv(data.x, data.edge_index, data.edge_attr)
# edge_attr = filter_out_expired_edge_attr(edge_attr)
# print("HI", x, edge_attr)
# #%%
# x, edge_attr=conv(x, data.edge_index, edge_attr)
# edge_attr = filter_out_expired_edge_attr(edge_attr)
# print("HI 2 ",x, edge_attr)

# x, edge_attr=conv(x, data.edge_index, edge_attr)
# edge_attr = filter_out_expired_edge_attr(edge_attr)
# print("HI 3 ",x, edge_attr)


#%%
def add_initial_balance(node_list):
    return [[0, 0] for idx in node_list]

# the first run
x = add_initial_balance(node_list=node_list)
x = torch.tensor(x, dtype=torch.int)
edge_index, edge_attribute = calc_edges(df, date_list[0])
income_list = calc_income_list(edge_index)
conv = GCNconv(1,1)
x, edge_attr=conv(x, edge_index, edge_attribute)
edge_attr = filter_out_expired_edge_attr(edge_attr)
#%%
# remaining loops
# results = [x.data.cpu().numpy()[259]]
results = x.clone().detach() 
#%%
for mdate in date_list[1:]:
    
    edge_index_daily, edge_attribute_daily = calc_edges(df, mdate)
    if edge_attribute_daily.shape[1]!=0:
        edge_index = torch.cat([edge_index, edge_index_daily], axis=1)
        edge_attr = torch.cat([edge_attr, edge_attribute_daily], axis=1)
    income_list = calc_income_list(edge_index)

    x, edge_attr=conv(x, edge_index, edge_attr)
    edge_attr = filter_out_expired_edge_attr(edge_attr)
    results = torch.cat((results, x), dim=0)
    #break

#%%
# r = np.array(results)
r = torch.reshape(results, (results.shape[0]//x.shape[0],x.shape[0],x.shape[1]))
r = r.numpy()
balance = pd.DataFrame(r[:,:,0].T, columns=date_list) 
profit = pd.DataFrame(r[:,:,1].T, columns=date_list) 

#%%
balance.to_pickle('data/balance.pkl')
profit.to_pickle('data/profit.pkl')
























# %%
