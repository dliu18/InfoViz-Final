import numpy as np
import math
import networkx as nx
from sklearn.manifold import TSNE
from umap import UMAP
import plotly.express as px
import plotly.graph_objects as go

import time

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



def draw_network(G, scores, title="Local Graph Topology",
                selection_local = None, selectedpoints = None):

    nodePos = nx.spring_layout(G, seed=42) # added seed argument for layout reproducibility

    # draw network topology
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = nodePos[edge[0]]
        x1, y1 = nodePos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = nodePos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        selectedpoints=selectedpoints,
        unselected={'marker': { 'opacity': 0.3 }},
        selected={'marker': { 'opacity': 1 }},
        marker=dict(
            showscale=False,
            colorscale='Reds',
            reversescale=True,
            size=10,
            line_width=2))

    # score encoding
    node_text = ['unfairness score: {}'.format(round(s,2)) for s in scores]
    node_trace.marker.color = scores
    node_trace.text = node_text

    # configure viz layout
    fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=title,
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    dragmode='select',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    # draw brush
    if selection_local and selection_local['range']:
        ranges = selection_local['range']
        selection_bounds = {'x0': ranges['x'][0], 'x1': ranges['x'][1],
                            'y0': ranges['y'][0], 'y1': ranges['y'][1]}
        bound_width = 1
    else:
        selection_bounds = {'x0': 0, 'x1': 0,
                            'y0': 0, 'y1': 0}
        bound_width = 0

    fig.add_shape(dict({'type': 'rect',
                        'line': { 'width': bound_width,'dash': 'dot','color': 'darkgrey'} },
                       **selection_bounds))

    return fig




def draw_embedding_2dprojection(projections, scores, type="TSNE",
                selection_local = None, selectedpoints = None):

    # embedding = np.load(embedding_dir)
    # G = nx.read_edgelist(graph_dir)
    # W = nx.to_numpy_array(G)

    # if type=="TSNE":
    #     tsne = TSNE(n_components=2, random_state=0)
    #     projections = tsne.fit_transform(embedding)
    # elif type=="UMAP":
    #     umap_2d = UMAP(n_components=2, init='random', random_state=0)
    #     projections = umap_2d.fit_transform(embedding)

    mark_trace = go.Scatter(
        x=list(projections[0]), y=list(projections[1]),
        mode='markers',
        hoverinfo='text',
        selectedpoints=selectedpoints,
        unselected={'marker': { 'opacity': 0.3 }},
        selected={'marker': { 'opacity': 1 }},
        marker=dict(
            showscale=True,
            colorscale='Reds',
            reversescale=True,
            size=10,
            colorbar=dict(
                thickness=15,
                title='Unfairness scores',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    # embedding = np.load(embedding_dir)
    # scores = unfairness_scores_normalized(embedding, W, G)
    # scores_unnormalized = unfairness_scores(embedding, W)
    mark_text = ['unfairness score: {}'.format(round(s,2)) for s in scores]
    mark_trace.marker.color = scores
    mark_trace.text = mark_text

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
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=True, zeroline=False, showticklabels=True),
                    yaxis=dict(showgrid=True, zeroline=False, showticklabels=True))
                    )

    if selection_local and selection_local['range']:
        ranges = selection_local['range']
        selection_bounds = {'x0': ranges['x'][0], 'x1': ranges['x'][1],
                            'y0': ranges['y'][0], 'y1': ranges['y'][1]}
        bound_width = 1
    else:
        selection_bounds = {'x0': 0, 'x1': 0,
                            'y0': 0, 'y1': 0}
        bound_width = 0

    fig.add_shape(dict({'type': 'rect',
                        'line': { 'width': bound_width,'dash': 'dot','color': 'darkgrey'} },
                       **selection_bounds))

    return fig



def load_network(path, embedding_dir, title="Local Graph Topology",
                selection_local = None, selectedpoints = None):

    G = nx.read_edgelist(path)
    W = nx.to_numpy_array(G)
    nodePos = nx.spring_layout(G, seed=42) # added seed argument for layout reproducibility

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = nodePos[edge[0]]
        x1, y1 = nodePos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = nodePos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        selectedpoints=selectedpoints,
        unselected={'marker': { 'opacity': 0.3 }},
        selected={'marker': { 'opacity': 1 }},
        marker=dict(
            showscale=False,
            colorscale='Reds',
            reversescale=True,
            size=10,
            line_width=2))

    embedding = np.load(embedding_dir)
    scores = unfairness_scores_normalized(embedding, W, G)
    scores_unnormalized = unfairness_scores(embedding, W)
    node_text = ['unfairness score: {}'.format(round(s,2)) for s in scores]
    node_trace.marker.color = scores
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=title,
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    dragmode='select',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    if selection_local and selection_local['range']:
        ranges = selection_local['range']
        selection_bounds = {'x0': ranges['x'][0], 'x1': ranges['x'][1],
                            'y0': ranges['y'][0], 'y1': ranges['y'][1]}
        bound_width = 1
    else:
        selection_bounds = {'x0': 0, 'x1': 0,
                            'y0': 0, 'y1': 0}
        bound_width = 0

    fig.add_shape(dict({'type': 'rect',
                        'line': { 'width': bound_width,'dash': 'dot','color': 'darkgrey'} },
                       **selection_bounds))

    return fig




def load_embedding_2dprojection_go(embedding_dir, graph_dir, type="TSNE",
                selection_local = None, selectedpoints = None):

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
        selectedpoints=selectedpoints,
        unselected={'marker': { 'opacity': 0.3 }},
        selected={'marker': { 'opacity': 1 }},
        marker=dict(
            showscale=True,
            colorscale='Reds',
            reversescale=True,
            size=10,
            colorbar=dict(
                thickness=15,
                title='Unfairness scores',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    embedding = np.load(embedding_dir)
    scores = unfairness_scores_normalized(embedding, W, G)
    scores_unnormalized = unfairness_scores(embedding, W)
    mark_text = ['unfairness score: {}'.format(round(s,2)) for s in scores]
    mark_trace.marker.color = scores
    mark_trace.text = mark_text

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
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=True, zeroline=False, showticklabels=True),
                    yaxis=dict(showgrid=True, zeroline=False, showticklabels=True))
                    )

    if selection_local and selection_local['range']:
        ranges = selection_local['range']
        selection_bounds = {'x0': ranges['x'][0], 'x1': ranges['x'][1],
                            'y0': ranges['y'][0], 'y1': ranges['y'][1]}
        bound_width = 1
    else:
        selection_bounds = {'x0': 0, 'x1': 0,
                            'y0': 0, 'y1': 0}
        bound_width = 0

    fig.add_shape(dict({'type': 'rect',
                        'line': { 'width': bound_width,'dash': 'dot','color': 'darkgrey'} },
                       **selection_bounds))

    return fig




def get_statistical_summary(G):
    '''computes the statistical summary of the graph G'''
    n = nx.number_of_nodes(G)
    m = nx.number_of_edges(G)
    density = nx.density(G)
    number_of_triangles = int(sum(nx.triangles(G).values()) / 3)
    avg_clust_coeff = nx.average_clustering(G)
    return n,m,density,number_of_triangles,avg_clust_coeff


def get_egoNet(G, node, k=1):
    '''Returns the k-hop ego net of a node, from node index.
    k: max distance of neighbors from node.'''
    tic = time.perf_counter()

    ego_net = nx.ego_graph(G, node, radius=k)

    toc = time.perf_counter()
    print(f"Calculated the ego net in {toc - tic:0.4f} seconds")

    return ego_net,[int(node_id) for node_id in ego_net.nodes]

def get_induced_subgraph(G, node_list):
    return G.subgraph(node_list)
