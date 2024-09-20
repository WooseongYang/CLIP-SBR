import pandas as pd
import networkx as nx
import igraph as ig
import leidenalg

def create_session_graph(train_data):
    df = pd.DataFrame()
    df['session_id'] = train_data.dataset['session_id'].numpy()
    df['item_id'] = train_data.dataset['item_id'].numpy()
    
    # Create a graph using NetworkX
    G_nx = nx.Graph()

    # Add edges with weights
    for session_id, group in df.groupby('session_id'):
        items = group['item_id'].tolist()
        for i in range(len(items) - 1):
            # If edge already exists, increase weight by 1
            if G_nx.has_edge(items[i], items[i + 1]):
                G_nx[items[i]][items[i + 1]]['weight'] += 1
            else:
                G_nx.add_edge(items[i], items[i + 1], weight=1)
    
    return G_nx

def detect_communities(G_nx, resolution):
    # Convert NetworkX graph to an igraph graph with weights
    G_ig = ig.Graph.TupleList(G_nx.edges(data='weight'), directed=False, edge_attrs=['weight'])

    # Apply the Leiden algorithm with consideration of edge weights
    partition = leidenalg.find_partition(
        G_ig, 
        partition_type=leidenalg.RBConfigurationVertexPartition, 
        resolution_parameter=resolution, 
        weights='weight'
    )

    # Create a mapping from node names to community IDs
    partition_dict = {int(node['name']): community_id + 1 for node, community_id in zip(G_ig.vs, partition.membership)}
    return partition_dict

def add_edges(G_nx, data):
    df = pd.DataFrame()
    df['session_id'] = data.dataset['session_id'].numpy()
    df['item_id'] = data.dataset['item_id'].numpy()

    # Add edges for validation and test datasets
    for session_id, group in df.groupby('session_id'):
        items = group['item_id'].tolist()
        for i in range(len(items) - 1):
            # If edge already exists, increase weight by 1
            if G_nx.has_edge(items[i], items[i + 1]):
                G_nx[items[i]][items[i + 1]]['weight'] += 1
            else:
                G_nx.add_edge(items[i], items[i + 1], weight=1)
    return G_nx