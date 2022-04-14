# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import sys
import dash
from dash import Dash, html, dcc, Input, Output, callback, dash_table
import dash_bootstrap_components as dbc
import numpy as np
import networkx as nx
import pandas as pd

import time


sys.path.insert(0, './../')
from utilities_with_interaction import (unfairness_scores,
                        unfairness_scores_normalized,
                        load_network,
                        load_embedding_2dprojection_go,
                        get_statistical_summary,
                        get_egoNet,
                        get_induced_subgraph,
                        draw_network,
                        draw_embedding_2dprojection)


## Pre-Load Data
global G, df

graph_dir = 'edgelists/facebook_combined.edgelist'
embedding_dir = 'embeddings/Facebook/Node2Vec/Facebook_Node2Vec_64_embedding.npy'
preprocessed_data_dir = 'embeddings/Facebook/Node2Vec/Facebook_Node2Vec_64_embedding_node_features.csv'
G = nx.read_edgelist(graph_dir)
df = pd.read_csv(preprocessed_data_dir)

# !!! Here sort by score !!!
node_ids = df['id'].to_numpy()
scores = df[' InFoRM'].to_numpy()
focal_node = node_ids[np.argmax(scores)]
node_list_dict = {'Node IDs': node_ids, 'Scores': scores}
node_list = pd.DataFrame(data=node_list_dict)
n = nx.number_of_nodes(G)




## View Layout
app = Dash(__name__)
app.title = "InfoViz-Final"

diagnostics_layout = html.Div(
    children=[
        html.H1(children='Diagnostics view'),
        html.Div([
                    html.Div([
                        html.H3('Select an Embedding Algorithm:'),
                    ],style={'alignItems':'center','paddingRight': '10px','display': 'inline-block'}),
                    html.Div([
                        dcc.Dropdown(options=['Node2Vec','HGCN','HOPE','LaplacianEigenmap','SDNE','SVD'], 
                        value='Node2Vec', id='embeddingDropdown',style={'width':'200px','alignItems':'center'}),
                    ],style={'display': 'inline-block','verticalAlign': 'middle','paddingTop': '12px'})
                ],style=dict(display='flex')),
        html.Div([
                    html.Div([
                        html.H4('Select a Network:'),
                    ],style={'alignItems':'center','paddingRight': '10px','display': 'inline-block'}),
                    html.Div([
                        dcc.Dropdown(options=['Facebook'], value='Facebook', id='networkDropdown',style={'width':'160px','alignItems':'center'}),
                    ],style={'display': 'inline-block','verticalAlign': 'middle','paddingTop': '12px'}),
                    html.Div([
                        html.H4('Select a Projection Method:'),
                    ],style={'alignItems':'center', 'paddingLeft': '30px', 'paddingRight': '10px','display': 'inline-block'}),
                    html.Div([
                        dcc.Dropdown(options=['PCA'], value='PCA', id='projectionDropdown',style={'width':'160px','alignItems':'center'}),
                    ],style={'display': 'inline-block','verticalAlign': 'middle','paddingTop': '12px'})
                ],style=dict(display='flex')),
        html.Div(
            dbc.Container([
                dbc.Label('Click to select the focal node:'),
                dash_table.DataTable(node_list.to_dict('records'),
                                [{"name": i, "id": i} for i in node_list.columns], 
                                id='nodeList', 
                                style_table={'overflowY': 'auto'},
                                row_selectable='single',
                                sort_action="native",
                                page_size= min(10, n-1)
                                )
            ]),
            className='main view',
            style={'alignItems':'center', 'width': '15%', 'paddingRight': '20px', 'paddingBottom': '20px', 'display': 'inline-block'}
        ),
        html.Div(
            dcc.Graph(id='overviewDiagnosticsGraph', config={'displayModeBar': False}),
            className='main view',
            style={'width': '38%', 'display': 'inline-block'}
        ),
        html.Div(
            dcc.Graph(id='overviewDiagnosticsEmb', config={'displayModeBar': False}),
            className='main view',
            style={'width': '40%', 'display': 'inline-block'}
        ),
        html.Div([html.Button(children=[
                    html.A(href="/", children="Back")])])
    ]
)
app.layout = diagnostics_layout

    

## Handle Interaction
@callback(
    Output('overviewDiagnosticsGraph', 'figure'),
    Output('overviewDiagnosticsEmb', 'figure'),
    [Input('embeddingDropdown', 'value'),
     Input('networkDropdown', 'value'),
     Input('projectionDropdown', 'value'),
     Input('nodeList', 'selected_rows'),
     Input('overviewDiagnosticsGraph', 'selectedData'),
     Input('overviewDiagnosticsEmb', 'selectedData')]
)
def updateView(embeddingDropdown, networkDropdown, projectionDropdown,
                selectedRow, selectionGraph, selectionEmb):

    global G, df

    # Configure data sources
    graph_dir = 'edgelists/facebook_combined.edgelist'
    embedding_dir = 'embeddings/Facebook/{}/Facebook_{}_64_embedding.npy'.format(
                                                                            embeddingDropdown,
                                                                            embeddingDropdown)
    preprocessed_data_dir = 'embeddings/Facebook/{}/Facebook_{}_64_embedding_node_features.csv'.format(
                                                                                                embeddingDropdown,
                                                                                                embeddingDropdown)
 
    # Identify callback source
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    selection = False
    # UPDATE LOGIC:
    # 1) when we change network or embedding algo everything should be reset: 
    # clear brush and list selection 
    # 2) when we change projection method the brush should be released from the Emb only.
    if trigger_id == 'embeddingDropdown':
        # erase brushes
        selectionGraph = None
        selectionEmb = None
        # read preprocessed data
        df = pd.read_csv(preprocessed_data_dir)

    elif trigger_id == 'networkDropdown':
        # erase brushes
        selectionGraph = None
        selectionEmb = None
        # load new network 
        G = nx.read_edgelist(graph_dir)
        # read preprocessed data
        df = pd.read_csv(preprocessed_data_dir)
        
    elif trigger_id == 'projectionDropdown':
        # erase embedding brush
        selectionEmb = None
        projectionAlgo = projectionDropdown

    elif trigger_id == 'nodeList':
        # erase brushes
        selectionGraph = None
        selectionEmb = None
        # get new focal node
        if selectedRow:
            focalNodeIdx = selectedRow[0]

    else:
        pass

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


    # !!! Here sort by score !!!
    n = nx.number_of_nodes(G)
    if not selection:
        selectedpoints = np.array(range(n))
    node_ids = df['id'].to_numpy()
    scores = df[' InFoRM'].to_numpy()


    if selectedRow:
        focalNodeIdx = selectedRow[0]
        focal_node = node_ids[focalNodeIdx]
        print("\nSELECTED ROW = {}\n".format(selectedRow[0]))
        print("FOCAL NODE = {}\n".format(focal_node))
    else:
        focal_node = node_ids[np.argmax(scores)]
    node_list_dict = {'Node IDs': node_ids, 'Scores': scores}
    node_list = pd.DataFrame(data=node_list_dict)

    local_network, local_ids = get_egoNet(G, str(focal_node), k=1)
    local_scores = scores[local_ids]
    local_projections_x = df[' proj_x'].to_numpy()[local_ids]
    local_projections_y = df[' proj_x'].to_numpy()[local_ids]
    local_projections = np.vstack((local_projections_x,local_projections_y))

    tic = time.perf_counter()

    figGraph = draw_network(local_network, local_scores, title = "1-Hop Ego Network",
                                selection_local = selectionGraph, selectedpoints = selectedpoints)


    toc = time.perf_counter()

    print(f"Network drawn in {toc - tic:0.4f} seconds")

    figEmb = draw_embedding_2dprojection(local_projections, local_scores, type=projectionDropdown,
                                selection_local = selectionEmb, selectedpoints = selectedpoints)
    
    return figGraph, figEmb




if __name__ == '__main__':
    app.run_server(debug=True)