# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import sys
from dash import Dash, html, dcc, Input, Output, State, callback
import numpy as np
import networkx as nx
import pandas as pd
import plotly.express as px
import csv
import json
from dash.exceptions import PreventUpdate


sys.path.insert(0, './../')
from utilities import (unfairness_scores,
                        unfairness_scores_normalized,
                        load_network,
                        get_statistical_summary,
                        get_edgelist_file)

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
                        #dcc.Dropdown(options=['Example 1','Example 2'], value='Example 1', id='networkDropdown',style={'width':'160px','align-items':'center'}),
                        dcc.Dropdown(options=['Facebook','protein-protein', 'AutonomousSystems', 'ca-HepTh', 'LastFM', 'wikipedia'], value='Facebook', id='networkDropdown',style={'width':'160px','align-items':'center'}),
                    ],style={'display': 'inline-block','vertical-align': 'middle','padding-top': '12px'}),
                    html.Div([
                        html.H3('Select a fairness notion:'),
                    ],style={'align-items':'center','padding-right': '10px','display': 'inline-block', 'margin-left': '50px'}),
                    html.Div([
                        dcc.Dropdown(options=['Individual (InFoRM)','Group (Fairwalk)'], value='Individual (InFoRM)', id='fairnessNotion',style={'width':'200px','align-items':'center'}),
                    ],style={'display': 'inline-block','vertical-align': 'middle','padding-top': '12px'})
                ],style=dict(display='flex')
                ),
        html.Div([
            html.H3('Select fairness parameters'),
            html.Div(children=[],id='fairnessParams'), # en los children coloco todos los dropdowns para los hyperparametros
        ],style={'align-items':'center','padding-right': '10px','display': 'inline-block'}
                ),
        html.Div([
                    html.Div([
                        html.H4('Select an Embedding Algorithm:'),
                    ],style={'align-items':'center','padding-right': '10px','display': 'inline-block'}),
                    html.Div([
                        dcc.Dropdown(options=['Node2Vec','HOPE', 'HGCN', 'LaplacianEigenmap', 'SDNE', 'SVD'], value='Node2Vec', id='embDropdownLeft',style={'width':'200px','align-items':'center'}),
                    ],style={'display': 'inline-block','vertical-align': 'middle','padding-top': '12px'}),
                    html.Div([
                        html.H4('Select an Embedding Algorithm:'),
                    ],style={'align-items':'center', 'padding-left': '150px', 'padding-right': '10px','display': 'inline-block'}),
                    html.Div([
                        dcc.Dropdown(options=['Node2Vec','HOPE', 'HGCN', 'LaplacianEigenmap', 'SDNE', 'SVD'], value='Node2Vec', id='embDropdownRight', style={'width':'200px','align-items':'center'}),
                    ],style={'display': 'inline-block','vertical-align': 'middle','padding-top': '12px'})
                ],style=dict(display='flex')
                ),
        html.Div(
            children = [
                html.Div([
                    html.Div(id='overviewContainerLeft',style={}),
                ]),
                html.Div([html.Button(children=[
                    html.A(href="/diagnostics", children="Diagnose")])])
            ],
            className="title six columns",
            style={'width': '49%', 'display': 'inline-block'}
        ),
        html.Div(
            children = [
                html.Div(id='overviewContainerRight',style={}),
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
                        html.Div(id='overviewContainerStats1',
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
                    html.Div(id='overviewContainerStats2',style={'padding-bottom': '85px'}),
                ],
                className="title six columns",
                style={'width': '49%', 'display': 'inline-block'}
            )
        ])
    ]
)
app.layout = overview_layout

# callbacks

# callback for parameter selection depending on selected fairness notion
@callback(
    Output('fairnessParams', 'children'),
    [Input('networkDropdown', 'value'),
    Input('fairnessNotion', 'value')]
)
def display_fairness_parameters(networkDropdown, fairnessNotion):
    # print('display_fairness_parameters')
    # get path to selected network

    if fairnessNotion == 'Group (Fairwalk)':
        # get sensitive attributes
        config_file = "embeddings/{}/group_fairness_config.json".format(networkDropdown)
        with open(config_file, "r") as configFile:
            group_fairness_config = json.load(configFile)
        # verify case sensitive_attrs = []
        # get attribute values from the selected attribute
        params = html.Div([
                    html.Div([
                        html.Div([
                            html.H4('Select the number of hops:'),
                        ],style={'align-items':'center','padding-right': '10px', 'display': 'inline-block'}),
                        html.Div([
                            dcc.Dropdown(options=[1,2], value=1, id='nrHops',style={'width':'60px','align-items':'center'}),
                        ],style={'display': 'inline-block','vertical-align': 'middle','padding-top': '3px'}),
                    ],style={'display': 'none'}),
                    html.Div([
                        html.Div([
                            html.H4('Select a node attribute:'),
                        ],style={'align-items':'center','padding-right': '10px','display': 'inline-block'}),
                        html.Div([
                            dcc.Dropdown(options=group_fairness_config["sensitive_attrs"], value=group_fairness_config["sensitive_attrs"][0], id='sensitiveAttr',style={'width':'200px','align-items':'center'}),
                        ],style={'display': 'inline-block','vertical-align': 'middle','padding-top': '12px'}),
                        html.Div([
                            html.H4('Select an attribute value:'),
                        ],style={'align-items':'center', 'padding-left': '150px', 'padding-right': '10px','display': 'inline-block'}),
                        html.Div([
                            dcc.Dropdown(options=group_fairness_config["sensitive_attrs_vals"], value=group_fairness_config["sensitive_attrs_vals"][0], id='sensitiveAttrVal', style={'width':'200px','align-items':'center'}),
                        ],style={'display': 'inline-block','vertical-align': 'middle','padding-top': '12px'}),
                        html.Div([
                            html.H4('Select a value of k:'),
                        ],style={'align-items':'center', 'padding-left': '150px', 'padding-right': '10px','display': 'inline-block'}),
                        html.Div([
                            dcc.Dropdown(options=group_fairness_config["k_s"], value=group_fairness_config["k_s"][0], id='kVal', style={'width':'200px','align-items':'center'}),
                        ],style={'display': 'inline-block','vertical-align': 'middle','padding-top': '12px'})
                    ],style=dict(display='block')
                    )
                ])
    else:
        params = html.Div([
            html.Div([
                html.Div([
                    html.H4('Select the number of hops:'),
                ],style={'align-items':'center','padding-right': '10px', 'display': 'inline-block'}),
                html.Div([
                    dcc.Dropdown(options=[1,2], value=1, id='nrHops',style={'width':'60px','align-items':'center'}),
                ],style={'display': 'inline-block','vertical-align': 'middle','padding-top': '3px'}),
            ]),
            html.Div([
                html.Div([
                    html.H4('Select a node attribute:'),
                ],style={'align-items':'center','padding-right': '10px','display': 'inline-block'}),
                html.Div([
                    dcc.Dropdown(options=[0], value=0, id='sensitiveAttr',style={'width':'200px','align-items':'center'}),
                ],style={'display': 'inline-block','vertical-align': 'middle','padding-top': '12px'}),
                html.Div([
                    html.H4('Select an attribute value:'),
                ],style={'align-items':'center', 'padding-left': '150px', 'padding-right': '10px','display': 'inline-block'}),
                html.Div([
                    dcc.Dropdown(options=[0], value=0, id='sensitiveAttrVal', style={'width':'200px','align-items':'center'}),
                ],style={'display': 'inline-block','vertical-align': 'middle','padding-top': '12px'}),
                html.Div([
                    html.H4('Select a value of k:'),
                ],style={'align-items':'center', 'padding-left': '150px', 'padding-right': '10px','display': 'inline-block'}),
                html.Div([
                    dcc.Dropdown(options=[0], value=0, id='kVal', style={'width':'200px','align-items':'center'}),
                ],style={'display': 'inline-block','vertical-align': 'middle','padding-top': '12px'})
            ],style=dict(display='none')
            )
        ])
    return params

@callback(
    Output('overviewContainerLeft', 'children'),
    [State('networkDropdown', 'value'),
    State('embDropdownLeft', 'value'),
    State('fairnessNotion', 'value'),
    Input('sensitiveAttr', 'value'),
    Input('sensitiveAttrVal', 'value'),
    Input('kVal', 'value'),
    Input('nrHops', 'value')]
)
def update_network1_ind_fairness(networkDropdown, embDropdownLeft, fairnessNotion, sensitiveAttr, sensitiveAttrVal, kVal, nrHops):
    edgelist_file = "edgelists/{}".format(get_edgelist_file(networkDropdown))
    node_features_file = "embeddings/{}/{}/{}_{}_64_embedding_node_features.csv".format(networkDropdown, 
                                                                                            embDropdownLeft, 
                                                                                            networkDropdown, 
                                                                                            embDropdownLeft)
    
    path_fairness_score = ""
    if fairnessNotion == 'Individual (InFoRM)':
        params = {"nrHops": nrHops}
    else: #Group (Fairwalk)
        params = {"attribute": sensitiveAttr, "value": sensitiveAttrVal, "k": kVal}
        path_fairness_score = "embeddings/{}/{}/{}_{}_64_embedding_group_fairness_scores.csv".format(networkDropdown, 
                                                                                            embDropdownLeft, 
                                                                                            networkDropdown, 
                                                                                            embDropdownLeft)

    fig = load_network(edgelist_file, node_features_file, path_fairness_score, fairnessNotion, params)

    graph = dcc.Graph(
                    id='overview-graph-left',
                    figure=fig
                )
    return graph

@callback(
    Output('overviewContainerRight', 'children'),
    [State('networkDropdown', 'value'),
    State('embDropdownRight', 'value'),
    State('fairnessNotion', 'value'),
    Input('sensitiveAttr', 'value'),
    Input('sensitiveAttrVal', 'value'),
    Input('kVal', 'value'),
    Input('nrHops', 'value')]
)
def update_network2_ind_fairness(networkDropdown, embDropdownRight, fairnessNotion, sensitiveAttr, sensitiveAttrVal, kVal, nrHops):
    edgelist_file = "edgelists/{}".format(get_edgelist_file(networkDropdown))
    node_features_file = "embeddings/{}/{}/{}_{}_64_embedding_node_features.csv".format(networkDropdown, 
                                                                                            embDropdownRight, 
                                                                                            networkDropdown, 
                                                                                            embDropdownRight)
    
    path_fairness_score = ""
    if fairnessNotion == 'Individual (InFoRM)':
        params = {"nrHops": nrHops}
    else: #Group (Fairwalk)
        params = {"attribute": sensitiveAttr, "value": sensitiveAttrVal, "k": kVal}
        path_fairness_score = "embeddings/{}/{}/{}_{}_64_embedding_group_fairness_scores.csv".format(networkDropdown, 
                                                                                            embDropdownRight, 
                                                                                            networkDropdown, 
                                                                                            embDropdownRight)

    fig = load_network(edgelist_file, node_features_file, path_fairness_score, fairnessNotion, params)

    graph = dcc.Graph(
                    id='overview-graph-right',
                    figure=fig
                )
    return graph

@callback(
    Output('nodes', 'children'),
    Output('edges', 'children'),
    Output('density', 'children'),
    Output('triangles', 'children'),
    Output('clustcoeff', 'children'),
    Output('overviewContainerStats2', 'children'),
    Input('networkDropdown', 'value')
)
def update_statistical_summary(networkDropdown):
    path = "edgelists/{}".format(get_edgelist_file(networkDropdown))


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
    # insert in Div
    return n,m,density,number_of_triangles,avg_clust_coeff, bar_chart
    

if __name__ == '__main__':
    app.run_server(debug=True)