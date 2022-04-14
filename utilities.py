import numpy as np
import pandas as pd
import math
import networkx as nx
from sklearn.manifold import TSNE
from umap import UMAP
import plotly.express as px
import plotly.graph_objects as go
import csv
import random

# Unfairness scores

def unfairness_score(Y, W, node_idx):
    '''
        Calculates the unfairness score for node 'node_idx' where Y is the nxd embedding matrix and W
        is the weighted adjacency matrix. 
        
        The unfairness score is \sum_{j=1}^n |Y_i - Y_j|^2 W[i,j]
    '''
    
    unfairness = 0.0
    for j in range(len(Y)):
        if W[node_idx][j] == 0:
            continue 
        unfairness += np.linalg.norm(Y[node_idx] - Y[j])*W[node_idx][j]
    return unfairness 

def unfairness_scores(Y, W):
    return [unfairness_score(Y, W, i) for i in range(len(Y))]

def unfairness_scores_normalized(Y, W, G):
    degrees = [G.degree[node] for node in G.nodes()]
    degree_normalized_scores = [unfairness_score(Y, W, i)/degrees[i] if degrees[i] > 0 
            else 0 
            for i in range(len(Y))]
    return degree_normalized_scores/np.max(degree_normalized_scores)

# group unfairness score

def recommended_nodes(Y,W,node_idx,k):
    '''
        Computes the set of recommended nodes for node 'node_idx' where Y is the nxd embedding matrix and W
        is the weighted adjacency matrix. 
        
        The set of recommended nodes is given by the top-$k$ most proximal ones in the embedding, using dot product similarity
    '''
    # compute top_k(<Y[u],Y[v]>)
    n = len(W)
    similarities = [(v,np.dot(Y[node_idx],Y[v])) for v in range(n) if (v != node_idx) and (not W[node_idx,v])]
    similarities.sort(key = lambda x : x[1],reverse=True)
    top_k = similarities[0:k]

    rho_u = [v for (v,_) in top_k]

    return rho_u      

def group_unfairness_score(Y, W, node_idx, node_features, S, z, k):
    '''
        Calculates the group unfairness score for node 'node_idx' where Y is the nxd embedding matrix, W
        is the weighted adjacency matrix, S is a sensitive attribute and z an attribute value for S. 
        
        The group unfairness score is 1/|Z^S| - z-share(u) where:
            - Z^S is the set of all possible values of attribute S
            - z-share(u) = |rho_z(u)|/|rho(u)|
            - rho(u) is the set of recommended nodes
            - rho_z(u) = {v : v in rho(u) and attr(v,S)=z}
    '''
    rho_u = recommended_nodes(Y,W,node_idx,k)

    # get the number of values of attribute S
    nr_Svalues = len(np.unique(node_features[:,S]))

    # added list(node_features[:,0]).index(v) to avoid out of index  
    # accesses for nodes that do not have an S feature value
    # node_features is not ordered
    rho_u_z = [v for v in rho_u if v in list(node_features[:,0]) and node_features[list(node_features[:,0]).index(v),S] == z]  #  attr(v,S) == z

    if rho_u == []:
        # check if assigning 0 for this case makes sense
        z_share_u = 0
    else:
        z_share_u = len(rho_u_z)/len(rho_u)

    return 1/nr_Svalues - z_share_u

def group_unfairness_scores(Y, W, node_features, S, z, k):
    return [group_unfairness_score(Y, W, i, node_features, S, z, k) for i in range(len(Y))]

def load_network(path, path_node_features, path_fairness_scores, fairness_notion, params, title="Local Graph Topology"):
#def load_network(edgelist_file, node_features_file):

    G = nx.read_edgelist(path)
    W = nx.to_numpy_array(G)
    #node_features = np.loadtxt(open(path_node_features, "rb"), delimiter=",", skiprows=1).astype(int)

    # pos = nx.get_node_attributes(G2,'pos')

    # read in node features 
    node_features = {}
    with open(path_node_features, "r") as featuresCSV:
        #print(featuresCSV.read())
        features_lines = [line.strip().split(", ") for line in featuresCSV.readlines()]
        keys = features_lines[0]
        for i in range(1, len(features_lines)):
            single_node_features = {}
            for j in range(len(keys)):
                single_node_features[keys[j]] = features_lines[i][j]
            node_features[single_node_features["id"]] = single_node_features

    edge_x = []
    edge_y = []
    # for edge in G.edges():
    #     x0, y0 = float(node_features[edge[0]]["pos_x"]), float(node_features[edge[0]]["pos_y"])
    #     x1, y1 = float(node_features[edge[1]]["pos_x"]), float(node_features[edge[1]]["pos_y"])
    #     edge_x.append(x0)
    #     edge_x.append(x1)
    #     edge_x.append(None)
    #     edge_y.append(y0)
    #     edge_y.append(y1)
    #     edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = float(node_features[node]["pos_x"]), float(node_features[node]["pos_y"])
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='Reds',
            reversescale=True,
            color=[],
            size=5,
            #colorbar=dict(
            #    thickness=15,
            #    title='Unfairness scores',
            #    xanchor='left',
            #    titleside='right'
            #),
            line_width=1))

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append('# of connections: '+str(len(adjacencies[1])))

    #node_trace.marker.color = node_adjacencies
    #node_trace.text = node_text
    # distinguish fairness notion and parameters
    if fairness_notion == 'Individual (InFoRM)':
        scores = [float(node_features[nodeId]["InFoRM"]) for nodeId in G.nodes()]
    else:
        types = {"attribute": "str", "value": "str", "k": "str", "node_id": "str", "group_fairness_score": "float"}
        print(path_fairness_scores)
        group_fairness_pd = pd.read_csv(path_fairness_scores,
                                       index_col=["attribute", "value", "k", "node_id"],
                                       dtype=types)
        scores = [group_fairness_pd.loc[params["attribute"]]
                                    .loc[params["value"]]
                                    .loc[params["k"]]
                                    .loc[str(nodeId)]["group_fairness_score"] for nodeId in G.nodes()]

    node_text = ['unfairness score: {}'.format(round(s,2)) for s in scores]
    node_trace.marker.color = scores
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='Local Graph Topology',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="",
                        #text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig


def load_embedding_2dprojection(embedding_dir, graph_dir, type="TSNE"):

    embedding = np.load(embedding_dir)
    G = nx.read_edgelist(graph_dir)
    W = nx.to_numpy_array(G)

    if type=="TSNE":
        tsne = TSNE(n_components=2, random_state=0)
        projections = tsne.fit_transform(embedding)
    elif type=="UMAP":
        umap_2d = UMAP(n_components=2, init='random', random_state=0)
        projections = umap_2d.fit_transform(embedding)

    node_adjacencies = []
    node_text = []
    for _, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append('# of connections: '+str(len(adjacencies[1])))

    scores = unfairness_scores_normalized(embedding, W, G)
    scores_unnormalized = unfairness_scores(embedding, W)
    scores = [round(s,2) for s in scores]

    fig = px.scatter(
                projections, x=0, y=1,
                color=scores,
                color_continuous_scale='reds_r',
                labels={"0": "dimension 1", "1": "dimension 2", 'color': 'unfairness score'},
                title="2D projection of node embeddings ({})".format(type),
            )

    fig.update_layout(  dragmode='select')
    fig.update_traces(  marker=dict(  size=10,
                                    line=dict(  width=2
                                                #color='black'
                                            )),
                        selector=dict(mode='markers'))
                        #hovertemplate='unfairness score: {}'.format(round(scores,2)))

    return fig

def load_embedding_2dprojection_go(embedding_dir, graph_dir, type="TSNE"):

    embedding = np.load(embedding_dir)
    G = nx.read_edgelist(graph_dir)
    W = nx.to_numpy_array(G)

    if type=="TSNE":
        tsne = TSNE(n_components=2, random_state=0)
        projections = tsne.fit_transform(embedding)
    elif type=="UMAP":
        umap_2d = UMAP(n_components=2, init='random', random_state=0)
        projections = umap_2d.fit_transform(embedding)

    mark_trace = go.Scatter(
        x=list(projections.T[0]), y=list(projections.T[1]),
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='Reds',
            reversescale=True,
            color=[],
            size=10,
            #colorbar=dict(
            #    thickness=15,
            #    title='Unfairness scores',
            #    xanchor='left',
            #    titleside='right'
            #),
            line_width=2))

    embedding = np.load(embedding_dir)
    scores = unfairness_scores_normalized(embedding, W, G)
    scores_unnormalized = unfairness_scores(embedding, W)
    mark_text = ['unfairness score: {}'.format(round(s,2)) for s in scores]
    mark_trace.marker.color = scores
    mark_trace.text = mark_text

    go.Layout

    fig = go.Figure(data=mark_trace,
                layout=go.Layout(
                    title="2D projection of node embeddings ({})".format(type),
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    dragmode='select',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="",
                        #text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig

def get_statistical_summary(G):
    # computes the statistical summary of the graph G
    n = nx.number_of_nodes(G)
    m = nx.number_of_edges(G)
    density = nx.density(G)
    number_of_triangles = int(sum(nx.triangles(G).values()) / 3)
    avg_clust_coeff = nx.average_clustering(G)
    return n,m,density,number_of_triangles,avg_clust_coeff

def get_edgelist_file(networkName):
    name_to_edgelist = {"Facebook": "facebook_combined.edgelist",
                        "protein-protein": "ppi.edgelist",
                        "AutonomousSystems": "AS.edgelist",
                        "ca-HepTh": "ca-HepTh.edgelist",
                        "LastFM": "lastfm_asia_edges.edgelist",
                        "wikipedia": "wikipedia.edgelist"}
    if networkName in name_to_edgelist:
        return name_to_edgelist[networkName]
    else:
        return name_to_edgelist["Facebook"]