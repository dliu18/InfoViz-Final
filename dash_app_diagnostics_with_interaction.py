# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import sys
import dash
from dash import Dash, html, dcc, Input, Output, callback, dash_table
import dash_bootstrap_components as dbc
import flask
import numpy as np
import networkx as nx
import pandas as pd
import json

import time
from urllib import parse


from utilities_with_interaction import (get_egoNet,
                                        get_induced_subgraph,
                                        draw_network,
                                        draw_embedding_2dprojection,
                                        get_scores,
                                        get_node_features)

## Define metadata 
graph_metadata = {"Facebook": {"edgelist": "edgelists/facebook_combined.edgelist"},
                 "LastFM": {"edgelist": "edgelists/lastfm_asia_edges.edgelist"},
                 "wikipedia": {"edgelist": "edgelists/wikipedia.edgelist"},
                  "protein-protein": {"edgelist": "edgelists/ppi.edgelist"},
                  "ca-HepTh": {"edgelist": "edgelists/ca-HepTh.edgelist"},
                  "AutonomousSystems": {"edgelist": "edgelists/AS.edgelist"},
                 }




## Pre-Load Data
global G, node_features, scores

networkDefault = "Facebook"
embDefault = "Node2Vec"

graph_dir = graph_metadata[networkDefault]["edgelist"]
preprocessed_data_dir = 'embeddings/{}/{}/{}_{}_64_embedding_node_features_InFoRM_scores.csv'.format(networkDefault,
                                                                                                        embDefault,
                                                                                                        networkDefault,
                                                                                                        embDefault)
preprocessed_group_fairness_dir = 'embeddings/{}/{}/{}_{}_64_embedding_group_fairness_scores.csv'.format(networkDefault,
                                                                                                            embDefault,
                                                                                                            networkDefault,
                                                                                                            embDefault)

G = nx.read_edgelist(graph_dir)
node_features = get_node_features(preprocessed_data_dir)
params = {"nrHops": 1}
scores_raw = get_scores('Individual (InFoRM)', params, preprocessed_data_dir)
scores = np.array(scores_raw).round(decimals=2)


## Import descriptive text
sys.path.insert(0, "description-txt/")
import diagnostics_description

## View Layout
app = Dash(__name__)
app.title = "InfoViz-Final"

diagnostics_layout = html.Div(
    children=[
        diagnostics_description.description,
        html.Div([
                    html.Div([
                        html.H3('Select an Embedding Algorithm:'),
                    ],style={'alignItems':'center','paddingRight': '10px','paddingTop': '7px','display': 'inline-block'}),
                    html.Div([
                        dcc.Dropdown(options=['Node2Vec','HGCN','HOPE','LaplacianEigenmap','SDNE','SVD'], 
                        value='Node2Vec', id='embeddingDropdown', clearable=False, style={'width':'200px','alignItems':'center'}),
                    ],style={'display': 'inline-block','verticalAlign': 'middle','paddingRight': 10})
                ],style=dict(display='flex')),
        html.Div([
                    html.Div([
                        html.H3('Select a Network:'),
                    ],style={'alignItems':'center','paddingRight': '10px','paddingTop': '7px','display': 'inline-block'}),
                    html.Div([
                        dcc.Dropdown(options=['Facebook','protein-protein', 'LastFM', 'wikipedia'], value='Facebook', id='networkDropdown', clearable=False, style={'width':'160px','alignItems':'center'}),
                    ],style={'display': 'inline-block','verticalAlign': 'middle','paddingTop': '0px'}),
                    html.Div([
                        html.H3('Select a fairness notion:'),
                        ],style={'alignItems':'center','paddingRight': '10px','paddingTop': '7px','display': 'inline-block', 'marginLeft': '50px'}),
                    html.Div([
                        dcc.Dropdown(options=[], value='Individual (InFoRM)', id='fairnessNotion_d', clearable=False, style={'width':'200px','alignItems':'center'}),
                        ],style={'display': 'inline-block','verticalAlign': 'middle','paddingTop': '0px'}),
                    html.Div([
                        html.H4('Select a Projection Method:'),
                    ],style={'alignItems':'center', 'paddingLeft': '30px', 'paddingRight': '10px','display': 'none'}),
                    html.Div([
                        dcc.Dropdown(options=['PCA'], value='PCA', id='projectionDropdown', clearable=False, style={'width':'160px','alignItems':'center'}),
                    ],style={'display': 'none','padding': 20, 'verticalAlign': 'middle'})
                ],style=dict(display='flex')),
        html.Div([
            html.H3('Select fairness parameters'),
            # div for fairness parameters - initialization
            html.Div(children=[
                html.Div([
                    html.Div([
                            html.Div([
                                html.H4('Select the number of hops:'),
                            ],style={'display': 'inline-block','alignItems':'center','paddingRight': '10px'}),
                            html.Div([
                                dcc.Dropdown(options=[], value='', id='nrHops_d', clearable=False, style={'width':'60px','alignItems':'center'}),
                            ],style={'display': 'inline-block','verticalAlign': 'middle','paddingTop': '3px'}),
                        ],id='indFairnessParams_d',style={'display': 'none'}),
                    html.Div([
                            html.Div([
                                html.H4('Select a node attribute:'),
                            ],style={'alignItems':'center','paddingRight': '10px','display': 'inline-block'}),
                            html.Div([
                                dcc.Dropdown(options=[], value='', id='sensitiveAttr_d', clearable=False, style={'width':'200px','alignItems':'center'}),
                            ],style={'display': 'inline-block','verticalAlign': 'middle','paddingTop': '0px'}),
                            html.Div([
                                html.H4('Select an attribute value:'),
                            ],style={'alignItems':'center', 'paddingLeft': '150px', 'paddingRight': '10px','display': 'inline-block'}),
                            html.Div([
                                dcc.Dropdown(options=[], value='', id='sensitiveAttrVal_d', clearable=False, style={'width':'200px','alignItems':'center'}),
                            ],style={'display': 'inline-block','verticalAlign': 'middle','paddingTop': '0px'}),
                            html.Div([
                                html.H4('Select a value of k:'),
                            ],style={'alignItems':'center', 'paddingLeft': '150px', 'paddingRight': '10px','display': 'inline-block'}),
                            html.Div([
                                dcc.Dropdown(options=[], value='', id='kVal_d', clearable=False, style={'width':'200px','alignItems':'center'}),
                            ],style={'display': 'inline-block','verticalAlign': 'middle','paddingTop': '0px'})
                        ],id='groupFairnessParams_d',style={'display': 'none'})
                    ])
                ],id='fairnessParams_d'),
            ],style={'alignItems':'center','padding': 20,'display': 'inline-block'}),
        html.Div(id='legend-indfairness',
            children=[
                html.H2('Legend'),
                html.H3(children=['The focal node is highlighted in ',html.Span('red',style={"backgroundColor":"#de2d26","color":"#ededed"}),' and increased in ', html.Span('size',style={'fontSize': 24}),'.']),
                html.H3(children=['The 1-hop neighbors of the focal node are colored in ',html.Span('coral',style={"backgroundColor":"#fcae91"}) ,', while the 2-hop neighbors in ',html.Span('light salmon',style={"backgroundColor":"#fee5d9"}),'.'])
            ],
            style={'display': 'block','paddingLeft':'350px','paddingBottom':'25px'}),
        html.Div(id='legend-groupfairness',
            children=[
                html.H2('Legend'),
                html.H3(children=['The focal node is increased in ',html.Span('size',style={'fontSize': 24}),' and its contour is highlighted in ',html.Span('red',style={"backgroundColor":"#de2d26","color":"#ededed"}),'.']),
                html.H3(children=['The 1-hop neighbors of the focal node, together with itself, are colored according to their gender value: ', html.Span('yellow',style={"backgroundColor":"#ffff99"}),' for gender 0 and ',html.Span('blue',style={"backgroundColor":"#386cb0","color":"#ededed"}),' for gender 1.'])
            ],
            style={'display': 'none','paddingLeft':'350px','paddingBottom':'25px'}),
        html.Div([
            html.Div(
                dbc.Container([
                    dbc.Label('Click to select the focal node:'),
                    dash_table.DataTable(data = [],
                                    id='nodeList', 
                                    style_table={'overflowY': 'auto'},
                                    row_selectable='single',
                                    sort_action="native",
                                    page_size= 10
                                    )
                ]),
                className='main view',
                style={'alignItems':'center', 'width': '15%', 'paddingRight': '20px', 'paddingBottom': '20px', 'display': 'inline-block'}
            ),
            html.Div(
                dcc.Loading(
                id="loading-graph",
                type="circle",
                color="#800020",
                children= dcc.Graph(id='overviewDiagnosticsGraph', config={'displayModeBar': False}),
                ),
                className='main view',
                style={'width': '40%', 'display': 'inline-block'}
            ),
            html.Div(
                dcc.Loading(
                id="loading-emb",
                type="circle",
                color="#800020",
                children= dcc.Graph(id='overviewDiagnosticsEmb', config={'displayModeBar': False}),
                ),
                className='main view',
                style={'width': '40%', 'display': 'inline-block'}       
            ),
        ],style=dict(display='flex')),
        html.Div(
            children = html.Div([html.Center(html.A(href="/", children="Back", className="button"))]),
            className='main view',
            style={'width': '10%', 'display': 'inline-block'})
    ]
)
app.layout = diagnostics_layout



## Callbacks

# # parse url
# @app.callback(
#     Output('networkDropdown', 'value'),
#     Output('embeddingDropdown', 'value'),
#     [Input('url', 'pathname')])
# def callback_func(url):
#     ## Get default parameters
#     print(url)
#     default_params = dict(parse.parse_qsl(parse.urlsplit(url).query))
#     networkDefault = default_params['net']
#     embDefault = default_params['emb']
#     print(networkDefault, embDefault)
#     return networkDefault, embDefault

# callback for parameter selection depending on selected fairness notion
@callback(
    Output('fairnessNotion_d', 'options'),
    Output('fairnessNotion_d', 'value'),
    # individual fairness
    Output('nrHops_d', 'options'),
    Output('nrHops_d', 'value'),
    Output('indFairnessParams_d', 'style'),
    # group fairness
    Output('sensitiveAttr_d', 'options'),
    Output('sensitiveAttr_d', 'value'),
    Output('sensitiveAttrVal_d', 'options'),
    Output('sensitiveAttrVal_d', 'value'),
    Output('kVal_d', 'options'),
    Output('kVal_d', 'value'),
    Output('groupFairnessParams_d', 'style'),
    [Input('networkDropdown', 'value'),
    Input('fairnessNotion_d', 'value')]
)
def display_fairness_parameters(networkDropdown, fairnessNotion):
    # print('display_fairness_parameters')
    # get path to selected network
    nr_hops_options = [] 
    nr_hops_options_value = '' 
    ind_fairness_params_style = {'display': 'none'}
    fairness_notions_val = fairnessNotion
    fairness_notions = []
    if networkDropdown == "Facebook":
        fairness_notions = ['Individual (InFoRM)','Group (Fairwalk)']
    else:
        fairness_notions = ['Individual (InFoRM)']
    
    ctx = dash.callback_context
    if ctx.triggered[0]['prop_id'] in ['networkDropdown.value','.'] :
        # restore state of parameters - default is Individual (InFoRM) with nr_hops = 1
        nr_hops_options = [1,2] 
        nr_hops_options_value = 1
        ind_fairness_params_style = {'display': 'inline-block','verticalAlign': 'middle','paddingTop': '3px'}
        fairness_notions_val = 'Individual (InFoRM)'
        group_fairness_params_style = {'display': 'none'}
        # group fairness
        sensitive_attr_options = []
        sensitive_attr_value = ''
        sensitive_attr_val_options = []
        sensitive_attr_val_value = ''
        k_val_options = []
        k_val_value = ''
        group_fairness_params_style = {'display': 'none'} 
    else:
        if fairnessNotion == 'Group (Fairwalk)':
            # get sensitive attributes
            config_file = "embeddings/{}/group_fairness_config.json".format(networkDropdown)
            with open(config_file, "r") as configFile:
                group_fairness_config = json.load(configFile)

            nr_hops_options = [] 
            nr_hops_options_value = '' 
            ind_fairness_params_style = {'display': 'none'}

            sensitive_attr_options = group_fairness_config["sensitive_attrs"]
            sensitive_attr_value = group_fairness_config["sensitive_attrs"][0]
            sensitive_attr_val_options = group_fairness_config["sensitive_attrs_vals"]
            sensitive_attr_val_value = group_fairness_config["sensitive_attrs_vals"][0]
            k_val_options = group_fairness_config["k_s"]
            k_val_value = group_fairness_config["k_s"][0]
            group_fairness_params_style = {'display': 'inline-block'}

        else:
            nr_hops_options = [1,2] 
            nr_hops_options_value = 1 
            ind_fairness_params_style = {'display': 'inline-block','verticalAlign': 'middle','paddingTop': '3px'}

            sensitive_attr_options = []
            sensitive_attr_value = ''
            sensitive_attr_val_options = []
            sensitive_attr_val_value = ''
            k_val_options = []
            k_val_value = ''
            group_fairness_params_style = {'display': 'none'}

    return fairness_notions, fairness_notions_val, nr_hops_options, nr_hops_options_value, ind_fairness_params_style, sensitive_attr_options, sensitive_attr_value, sensitive_attr_val_options, sensitive_attr_val_value, k_val_options, k_val_value, group_fairness_params_style



# Handle Interaction
@callback(
    Output('nodeList', 'data'),
    Output('overviewDiagnosticsGraph', 'figure'),
    Output('overviewDiagnosticsEmb', 'figure'),
    Output('legend-indfairness', 'style'),
    Output('legend-groupfairness', 'style'),
    [Input('embeddingDropdown', 'value'),
     Input('networkDropdown', 'value'),
     Input('projectionDropdown', 'value'),
     # brushing & linking and filtering selections
     Input('nodeList', 'selected_rows'),
     Input('overviewDiagnosticsGraph', 'selectedData'),
     Input('overviewDiagnosticsEmb', 'selectedData'),
     # fairness config
     Input('fairnessNotion_d', 'value'),
     Input('sensitiveAttr_d', 'value'),
     Input('sensitiveAttrVal_d', 'value'),
     Input('kVal_d', 'value'),
     Input('nrHops_d', 'value')
    ]
)
def updateView(embeddingDropdown, networkDropdown, projectionDropdown,
                selectedRow, selectionGraph, selectionEmb,
                fairnessNotion, sensitiveAttr, sensitiveAttrVal, kVal, nrHops):

    global G, node_features, scores

    legend_ind_fairness = {}
    legend_group_fairness = {}

    if fairnessNotion=='Group (Fairwalk)':
        print("Debug: attr = {}  attr_val = {}  #recommendation = {}\n".format(sensitiveAttr, 
                                                                                sensitiveAttrVal,
                                                                                kVal))
        attribute_type = sensitiveAttr
        hops = 1
        legend_ind_fairness = {'display': 'none'}
        legend_group_fairness = {'display': 'block','paddingLeft':'350px','paddingBottom':'25px'}
    else: #'Individual (InFoRM)'
        print("Debug: #hops = {}\n".format(nrHops))
        attribute_type = "distance"
        hops = int(nrHops)
        legend_group_fairness = {'display': 'none'}
        legend_ind_fairness = {'display': 'block','paddingLeft':'350px','paddingBottom':'25px'}

    # Configure data sources
    graph_dir = graph_metadata[networkDropdown]["edgelist"]
    preprocessed_data_dir = 'embeddings/{}/{}/{}_{}_64_embedding_node_features_InFoRM_scores.csv'.format(  networkDropdown,
                                                                                                embeddingDropdown,
                                                                                                networkDropdown,
                                                                                                embeddingDropdown)
    preprocessed_group_fairness_dir = 'embeddings/{}/{}/{}_{}_64_embedding_group_fairness_scores.csv'.format(  
                                                                                                networkDropdown,
                                                                                                embeddingDropdown,
                                                                                                networkDropdown,
                                                                                                embeddingDropdown)
    
    # Identify callback source
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # UPDATE LOGIC:
    # 1) when we change network or embedding algo everything should be reset: 
    # clear brush and list selection 
    # 2) when we change projection method the brush should be released from the Emb only.
    if trigger_id == 'networkDropdown':
        # erase brushes
        selectionGraph = None
        selectionEmb = None
        # load new network 
        G = nx.read_edgelist(graph_dir)
        # read preprocessed data
        node_features = get_node_features(preprocessed_data_dir)
        # must reload scores
        load_scores = True

    elif trigger_id == 'embeddingDropdown':
        # erase brushes
        selectionGraph = None
        selectionEmb = None
        # read preprocessed data
        node_features = get_node_features(preprocessed_data_dir)
        # must reload scores
        load_scores = True

    elif trigger_id == 'projectionDropdown':
        # erase embedding brush
        selectionEmb = None
        # no need to reload scores
        load_scores = False
        # get projection method
        projectionAlgo = projectionDropdown

    elif trigger_id == 'nodeList':
        # erase brushes
        selectionGraph = None
        selectionEmb = None
        # no need to reload scores
        load_scores = False
        # get new focal node
        if selectedRow:
            focalNodeIdx = selectedRow[0]

    elif trigger_id in ['fairnessNotion_d', 'sensitiveAttr_d', 'sensitiveAttrVal_d', 'kVal_d', 'nrHops_d']:
        # must reload scores for any change in the fairness config 
        print("got the fairness change " + trigger_id)
        load_scores = True
    
    else:
        # no need to reload scores
        load_scores = False

    # load scores, when needed
    if load_scores:
        if fairnessNotion=='Group (Fairwalk)':
            params = {"attribute": sensitiveAttr, "value": sensitiveAttrVal, "k": kVal}
            scores_raw = get_scores(fairnessNotion, params, preprocessed_group_fairness_dir)
        else: #'Individual (InFoRM)'
            params = {"nrHops": nrHops}
            scores_raw = get_scores(fairnessNotion, params, preprocessed_data_dir)
        scores = np.array(scores_raw).round(decimals=2)

    # handle selection triggers
    selection = False
    if trigger_id == 'overviewDiagnosticsEmb':
        # erase graph brush
        selectionGraph = None
        # get new selection
        if selectionEmb and selectionEmb['points']:
            selectedpoints = np.array([p['pointIndex'] for p in selectionEmb['points']])
            selection = True

    elif selectionGraph or trigger_id == 'overviewDiagnosticsGraph':
        # erase embedding brush
        selectionEmb = None
        # get new selection
        if selectionGraph and selectionGraph['points']:
            selectedpoints = np.array([p['pointIndex'] for p in selectionGraph['points']])
            selection = True

    else:
        selectionGraph = None
        selectionEmb = None


    node_ids = np.array(list(G.nodes()))
    n = nx.number_of_nodes(G)
    print("Number of nodes: ", n)
    if not selection:
        selectedpoints = np.array(range(n))

    # get focal node
    if selectedRow:
        focalNodeIdx = selectedRow[0]
        focal_node = node_ids[focalNodeIdx]
        print("\nSELECTED ROW = {}\n".format(selectedRow[0]))
        print("FOCAL NODE = {}\n".format(focal_node))
    else:
        focal_node = node_ids[np.argmax(scores)]
    node_list_dict = {'Node IDs': node_ids, 'Scores': scores}
    node_list = pd.DataFrame(data=node_list_dict)

    # get local topology and attributes
    local_network, local_ids = get_egoNet(G, str(focal_node), k=hops)
    if fairnessNotion=='Group (Fairwalk)':
        attributes = [int(node_features[idx][attribute_type]) for idx in local_ids]
        #attributes = [0 if idx==focal_node else 1 for idx in local_ids]
    else: #'Individual (InFoRM)'
        distance_dict = nx.shortest_path_length(local_network, source=focal_node)
        attributes = [0 if n==focal_node else distance_dict[n] for n in local_network]
    # focal node(s) list
    focal = [1 if idx==focal_node else 0 for idx in local_ids]

    local_scores = scores[[int(idx) for idx in local_ids]]
    local_projections_x = np.array([float(node_features[idx]['proj_x']) for idx in local_ids])
    local_projections_y = np.array([float(node_features[idx]['proj_y']) for idx in local_ids])
    local_projections = np.vstack((local_projections_x,local_projections_y))

    tic = time.perf_counter()

    figGraph = draw_network(local_network, local_scores, focal, fairness_notion=fairnessNotion, 
                        attributes=attributes, attribute_type=attribute_type,
                        title = "{}-Hop Ego Network".format(hops),
                        selection_local = selectionGraph, selectedpoints = selectedpoints)

    toc = time.perf_counter()

    print(f"Network drawn in {toc - tic:0.4f} seconds")

    figEmb = draw_embedding_2dprojection(local_network, local_projections, local_scores, focal, fairness_notion=fairnessNotion, 
                                attributes=attributes, attribute_type=attribute_type,
                                type=projectionDropdown,
                                selection_local = selectionEmb, selectedpoints = selectedpoints)
    


    return node_list.to_dict('records'), figGraph, figEmb, legend_ind_fairness, legend_group_fairness




if __name__ == '__main__':
    app.run_server(debug=True)