from dash import Dash, dcc, html, Input, Output, callback

import dash_app_overview
import dash_app_diagnostics_with_interaction

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/':
        return dash_app_overview.overview_layout
    elif pathname == '/diagnostics':
        return dash_app_diagnostics_with_interaction.diagnostics_layout
    else:
        return '404'

if __name__ == '__main__':
    app.run_server(debug=True)