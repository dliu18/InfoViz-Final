import numpy as np
import math
import networkx as nx
from sklearn.manifold import TSNE
from umap import UMAP
import plotly.express as px
import plotly.graph_objects as go


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

def load_network(path, embedding_dir):

    G = nx.read_edgelist(path)
    W = nx.to_numpy_array(G)
    # pos = nx.get_node_attributes(G2,'pos')

    # nodePos = nx.circular_layout(G)
    # nodePos = nx.kamada_kawai_layout(G)
    nodePos = nx.spring_layout(G, seed=42) # added seed argument for layout reproducibility
    # nodePos = nx.spectral_layout(G)

    # print(nodePos['0'])

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

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append('# of connections: '+str(len(adjacencies[1])))

    #node_trace.marker.color = node_adjacencies
    #node_trace.text = node_text

    embedding = np.load(embedding_dir)
    scores = unfairness_scores_normalized(embedding, W, G)
    scores_unnormalized = unfairness_scores(embedding, W)
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

    fig.update_traces(  marker=dict(  size=10,
                                    line=dict(  width=2
                                                #color='black'
                                            )),
                        selector=dict(mode='markers'))
                        #hovertemplate='unfairness score: {}'.format(round(scores,2)))

    return fig

def get_statistical_summary(G):
    # computes the statistical summary of the graph G
    n = nx.number_of_nodes(G)
    m = nx.number_of_edges(G)
    density = nx.density(G)
    number_of_triangles = int(sum(nx.triangles(G).values()) / 3)
    avg_clust_coeff = nx.average_clustering(G)
    return n,m,density,number_of_triangles,avg_clust_coeff