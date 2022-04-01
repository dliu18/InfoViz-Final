# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import sys
from dash import Dash, html, dcc, Input, Output, callback
import plotly.graph_objects as go
import numpy as np
import networkx as nx
import pandas as pd
import plotly.express as px

sys.path.insert(0, './../')
from utilities import (unfairness_scores,
                        unfairness_scores_normalized,
                        load_network,
                        get_statistical_summary)

app = Dash(__name__)
app.title = "InfoViz-Final"

overview_layout = html.Div(
    children=[
        html.H1(children='Overview'),
        html.Div([
                    html.Div([
                        html.H3('Select a Network:'),
                    ],style={'align-items':'center','padding-right': '10px','display': 'inline-block'}),
                    html.Div([
                        dcc.Dropdown(options=['Example 1','Example 2'], value='Example 1', id='demo-dropdown1',style={'width':'160px','align-items':'center'}),
                    ],style={'display': 'inline-block','vertical-align': 'middle','padding-top': '12px'})
                ],style=dict(display='flex')),
        html.Div([
                    html.Div([
                        html.H4('Select an Embedding Algorithm:'),
                    ],style={'align-items':'center','padding-right': '10px','display': 'inline-block'}),
                    html.Div([
                        dcc.Dropdown(options=['Embedding 1','Embedding 2'], value='Embedding 1', id='embedding-algo-1',style={'width':'200px','align-items':'center'}),
                    ],style={'display': 'inline-block','vertical-align': 'middle','padding-top': '12px'}),
                    html.Div([
                        html.H4('Select an Embedding Algorithm:'),
                    ],style={'align-items':'center', 'padding-left': '150px', 'padding-right': '10px','display': 'inline-block'}),
                    html.Div([
                        dcc.Dropdown(options=['Embedding 1','Embedding 2'], value='Embedding 2', id='embedding-algo-2', style={'width':'200px','align-items':'center'}),
                    ],style={'display': 'inline-block','vertical-align': 'middle','padding-top': '12px'})
                ],style=dict(display='flex')),
        html.Div(
            children = [
                html.Div([
                    html.Div(id='dd-output-container1-overview',style={}),
                ]),
                html.Div([html.Button(children=[
                    html.A(href="/diagnostics", children="Diagnose")])])
            ],
            className="title six columns",
            style={'width': '49%', 'display': 'inline-block'}
        ),
        html.Div(
            children = [
                html.Div(id='dd-output-container2-overview',style={}),
                html.Div([html.Button(children=[
                    html.A(href="/diagnostics", children="Diagnose")])])
            ],
            className="title six columns",
            style={'width': '49%', 'display': 'inline-block'}
        ),
        html.Div( children = [
            html.H2(children='Statistical summary of the network'),
            html.Div(
                children = [
                    html.Div([
                        html.Div(id='dd-output-container3',
                        children = [
                                html.Table([
                                    html.Tr([html.Th('Variable'), html.Th('Value')]),
                                    html.Tr([html.Td('n'), html.Td(id='nodes')]),
                                    html.Tr([html.Td('m'), html.Td(id='edges')]),
                                    html.Tr([html.Td('density'), html.Td(id='density')]),
                                    html.Tr([html.Td('nr. of triangles'), html.Td(id='triangles')]),
                                    html.Tr([html.Td('avg. clustering coeff.'), html.Td(id='clustcoeff')])
                                ], style={'width':'80%', 'border': '1px solid'})
                        ],style={'opacity':'1', 'padding-bottom': '180px'}),
                    ])
                ],
                className="title six columns",
                style={'width': '49%', 'display': 'inline-block'}
            ),
            html.Div(
                children = [
                    html.Div(id='dd-output-container4-overview',style={'padding-bottom': '85px'}),
                ],
                className="title six columns",
                style={'width': '49%', 'display': 'inline-block'}
            )
        ])
    ]
)
app.layout = overview_layout

# callbacks

@callback(
    Output('dd-output-container1-overview', 'children'),
    [Input('demo-dropdown1', 'value'),
    Input('embedding-algo-1', 'value')]
)
def update_network1(demodropdown1, embeddingalgo1):
    if embeddingalgo1 == "Embedding 1":
        embeddingAlgo = "embedding1"
    elif embeddingalgo1 == "Embedding 2":
        embeddingAlgo = "embedding2"
    else:
        embeddingAlgo = "embedding1"
    if demodropdown1 == "Example 1":
        graph_dir = 'data/example1.edgelist'
        embedding_dir = 'data2/example1/{}/example1_{}_10_embedding.npy'.format(embeddingAlgo,embeddingAlgo)
    elif demodropdown1 == "Example 2":
        graph_dir = 'data/example2.edgelist'
        embedding_dir = 'data2/example2/{}/example2_{}_10_embedding.npy'.format(embeddingAlgo,embeddingAlgo)
    else:
        # empty selection
        graph_dir = 'data/example1.edgelist'
        embedding_dir = 'data2/example1/embedding1/example1_embedding1_10_embedding.npy'
    fig = load_network(graph_dir, embedding_dir)
    graph = dcc.Graph(
                    id='example-graph-1',
                    figure=fig
                )
    return graph

@callback(
    Output('dd-output-container2-overview', 'children'),
    [Input('demo-dropdown1', 'value'),
    Input('embedding-algo-2', 'value')]
)
def update_network2(demodropdown1, embeddingalgo2):
    if embeddingalgo2 == "Embedding 1":
        embeddingAlgo = "embedding1"
    elif embeddingalgo2 == "Embedding 2":
        embeddingAlgo = "embedding2"
    else:
        embeddingAlgo = "embedding1"
    if demodropdown1 == "Example 1":
        graph_dir = 'data/example1.edgelist'
        embedding_dir = 'data2/example1/{}/example1_{}_10_embedding.npy'.format(embeddingAlgo,embeddingAlgo)
    elif demodropdown1 == "Example 2":
        graph_dir = 'data/example2.edgelist'
        embedding_dir = 'data2/example2/{}/example2_{}_10_embedding.npy'.format(embeddingAlgo,embeddingAlgo)
    else:
        # empty selection
        graph_dir = 'data/example1.edgelist'
        embedding_dir = 'data2/example1/embedding1/example1_embedding1_10_embedding.npy'
    fig = load_network(graph_dir, embedding_dir)
    graph = dcc.Graph(
                    id='example-graph-1',
                    figure=fig
                )
    return graph

@callback(
    Output('nodes', 'children'),
    Output('edges', 'children'),
    Output('density', 'children'),
    Output('triangles', 'children'),
    Output('clustcoeff', 'children'),
    Output('dd-output-container4-overview', 'children'),
    Input('demo-dropdown1', 'value')
)
def update_statistical_summary(value):
    if value == "Example 1":
        path = 'data/example1.edgelist'
    elif value == "Example 2":
        path = 'data/example2.edgelist'
    else:
        # empty selection
        path = 'data/example1.edgelist'
    # get graph
    G = nx.read_edgelist(path)
    # get properties
    n,m,density,number_of_triangles,avg_clust_coeff = get_statistical_summary(G)
    deg_hist = nx.degree_histogram(G)
    max_deg = len(deg_hist)
    deg_range = [i for i in range(0,max_deg)]
    df = pd.DataFrame(dict(degree=deg_range, nr_nodes=deg_hist))
    # create bar chart 
    fig = px.bar(df, x=df.degree, y=df.nr_nodes, labels={'degree' : 'Degree', 'nr_nodes':'Nr. of nodes'}, title='Degree distribution of the Network')
    fig.update_xaxes(type='category')
    bar_chart = dcc.Graph(figure=fig)
    # insert in Divs
    return n,m,density,number_of_triangles,avg_clust_coeff, bar_chart
    

if __name__ == '__main__':
    app.run_server(debug=True)