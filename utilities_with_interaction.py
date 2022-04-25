from enum import unique
from genericpath import exists
from platform import node
import numpy as np
import math
import networkx as nx
from sklearn.manifold import TSNE
from umap import UMAP
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import dash_table
import pandas


import time


def draw_network(G, scores, focal, fairness_notion='Individual (InFoRM)',
                attributes = None, attribute_type = None, 
                title="Local Graph Topology",
                selection_local = None, selectedpoints = None):

    nodePos = nx.spring_layout(G, seed=42) # added seed argument for layout reproducibility

    edge_x = []
    edge_y = []
    edge_lengths = []
    for edge in G.edges():
        x0, y0 = nodePos[edge[0]]
        x1, y1 = nodePos[edge[1]]
        length = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        edge_lengths.append(length)
    # for large and dense networks, show the longest 5% edges
    if nx.density(G)>0.6 and nx.number_of_nodes(G)>25: 
        edge_length_threshold = np.percentile(np.array(edge_lengths), 95)
    else:
        edge_length_threshold = np.min(np.array(edge_lengths))
    for edge in G.edges():
        x0, y0 = nodePos[edge[0]]
        x1, y1 = nodePos[edge[1]]
        length = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        if length >= edge_length_threshold:
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
            size=11,
            line_width=2))

    # focal node pop-out
    node_trace.marker.line['color'] = ['#de2d26' if f==1 else '#696969' for f in focal]
    node_trace.marker.line['width'] = [4 if f==1 else 2 for f in focal]
    node_trace.marker.size = [18 if f==1 else 11 for f in focal]

    # color encoding
    if attributes:  # if attributes exist, color by attribute
        # choose colormap
        if fairness_notion == 'Individual (InFoRM)':
            # sequencial: https://colorbrewer2.org/#type=sequential&scheme=Reds&n=5
            if len(set(attributes))==2: # for 1-hop 
                colormap = ['#fcae91','#a50f15']
            else: # for 2-hop or more
                colormap = ['#fee5d9','#fcae91','#a50f15']
        elif fairness_notion == 'Group (Fairwalk)':
            # categorical: https://colorbrewer2.org/#type=qualitative&scheme=Accent&n=7
            colormap = ['#386cb0', '#beaed4', '#7fc97f','#fdc086','#ffff99']
        node_trace.marker.color = attributes
        node_trace.marker.colorscale = colormap
        node_trace.marker.showscale = False
        node_trace.marker.reversescale = True
    else:   # else color by score
        # standardize coloscale
        if fairness_notion == 'Individual (InFoRM)':
            [val_min, val_max] = [0, 1]
        elif fairness_notion == 'Group (Fairwalk)':
            [val_min, val_max] = [-1, 1]
        else:
            [val_min, val_max] = [0, 1]
            node_trace.marker.color = scores
            node_trace.marker.colorscale = 'Reds'
            node_trace.marker.showscale = False
            node_trace.marker.cmin = val_min
            node_trace.marker.cmax = val_max

    # node info text
    if attributes:
        node_text = [" node id: {} <br> unfairness score: {} <br> {}: {}"
                    .format(n, scores[i], attribute_type, attributes[i]) 
                    for i,n in enumerate(G.nodes())]
    else:
        node_text = [" node id: {} <br> unfairness score: {}"
                    .format(n, scores[i]) 
                    for i,n in enumerate(G.nodes())]
    node_trace.text = node_text

    # configure viz layout
    fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=title,
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    dragmode='select',
                    uirevision=True,
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    fig.update_layout(legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    ))

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




def draw_embedding_2dprojection(G, projections, scores, focal, fairness_notion='Individual (InFoRM)',
                    attributes = None, attribute_type = None, 
                    type="TSNE",
                    selection_local = None, selectedpoints = None):

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
            size=11,
            line_width=2))

    # focal node pop-out
    mark_trace.marker.line['color'] = ['#de2d26' if f==1 else '#696969' for f in focal]
    mark_trace.marker.line['width'] = [4 if f==1 else 2 for f in focal]
    mark_trace.marker.size = [18 if f==1 else 11 for f in focal]

    # color encoding
    if attributes:  # if attributes exist, color by attribute
        # choose colormap
        if fairness_notion == 'Individual (InFoRM)':
            # sequencial: https://colorbrewer2.org/#type=sequential&scheme=Reds&n=5
            if len(set(attributes))==2: # for 1-hop 
                colormap = ['#fcae91','#a50f15']
            else: # for 2-hop or more
                colormap = ['#fee5d9','#fcae91','#a50f15']
        elif fairness_notion == 'Group (Fairwalk)':
            # categorical: https://colorbrewer2.org/#type=qualitative&scheme=Accent&n=7
            colormap = ['#386cb0', '#beaed4', '#7fc97f','#fdc086','#ffff99']
        mark_trace.marker.color = attributes
        mark_trace.marker.colorscale = colormap
        mark_trace.marker.showscale = False
        mark_trace.marker.reversescale = True
    else:   # else color by score
        # standardize coloscale
        if fairness_notion == 'Individual (InFoRM)':
            [val_min, val_max] = [0, 1]
        elif fairness_notion == 'Group (Fairwalk)':
            [val_min, val_max] = [-1, 1]
        else:
            [val_min, val_max] = [0, 1]
            mark_trace.marker.color = scores
            mark_trace.marker.colorscale = 'Reds'
            mark_trace.marker.showscale = False
            mark_trace.marker.cmin = val_min
            mark_trace.marker.cmax = val_max

    # node info text
    mark_text = [" node id: {} <br> unfairness score: {} <br> {}: {}"
                .format(n, scores[i], attribute_type, attributes[i]) 
                for i,n in enumerate(G.nodes())]
    mark_trace.text = mark_text

    min_x = np.abs(np.min(projections[0]))
    min_y = np.abs(np.min(projections[1]))
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
                    xaxis=dict(showgrid=True, zeroline=False, showticklabels=True, range=[np.min(projections[0])-min_x, np.max(projections[0])+min_x]),
                    yaxis=dict(showgrid=True, zeroline=False, showticklabels=True, range=[np.min(projections[1])-min_y, np.max(projections[1])+min_y])
                    )
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

def get_scores(fairness_notion, params, path_fairness_scores):
    # distinguish fairness notion and parameters
    if fairness_notion == 'Individual (InFoRM)':
        node_to_score = {}
        with open(path_fairness_scores, "r") as scores_file:
            lines = scores_file.readlines()
            header = lines[0].strip("\n").split(",")
            node_id_idx = header.index("id")
            nr_hops_idx = header.index("nr_hops")
            InFoRM_hops_idx = header.index("InFoRM_hops")
            for i in range(1, len(lines)):
                features = [feature.strip() for feature in lines[i].split(',')]
                if int(features[nr_hops_idx]) == params["nrHops"]:
                    try:
                        node_to_score[features[node_id_idx]] = float(features[InFoRM_hops_idx])
                    except:
                        print(features)
                        node_to_score[features[node_id_idx]] = 0.0
    else:
        node_to_score = {}
        with open(path_fairness_scores, "r") as scores_file:
            lines = scores_file.readlines()
            header = lines[0].strip("\n").split(",")
            node_id_idx = header.index("node_id")
            attribute_idx = header.index("attribute")
            value_idx = header.index("value")
            k_idx = header.index("k")
            group_fairness_score_idx = header.index("group_fairness_score")
            for i in range(1, len(lines)):
                features = [feature.strip() for feature in lines[i].split(',')]
                if features[attribute_idx] == params["attribute"] and\
                    features[value_idx] == params["value"] and\
                    features[k_idx] == params["k"]:
                    try:
                        node_to_score[features[node_id_idx]] = float(features[group_fairness_score_idx])
                    except:
                        print(features)
                        node_to_score[features[node_id_idx]] = 0.0

    scores = [score for i, (node_id, score) in enumerate(node_to_score.items())]

    return scores

def get_node_features(path_node_features):
    '''read in node features''' 
    node_features = {}
    with open(path_node_features, "r") as featuresCSV:
        features_lines = [line.strip().split(",") for line in featuresCSV.readlines()]
        keys = features_lines[0]
        for i in range(1, len(features_lines)):
            single_node_features = {}
            for j in range(len(keys)):
                single_node_features[keys[j]] = features_lines[i][j]
            node_features[single_node_features["id"]] = single_node_features

    return node_features

def get_egoNet(G, node, k=1):
    '''Returns the k-hop ego net of a node, from node index.
    k: max distance of neighbors from node.'''
    tic = time.perf_counter()

    ego_net = nx.ego_graph(G, node, radius=k)

    toc = time.perf_counter()
    print(f"Calculated the ego net in {toc - tic:0.4f} seconds")

    return ego_net,[idx for idx in ego_net.nodes()]

def get_induced_subgraph(G, node_list):
    return G.subgraph(node_list)
