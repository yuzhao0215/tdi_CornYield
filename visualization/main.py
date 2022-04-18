from dash import Dash, dcc, html, Input, Output
import plotly.express as px

import pandas as pd
import dash_bootstrap_components as dbc

from dash.dependencies import Input, Output
import plotly.graph_objects as go

df = pd.read_csv('total_data.csv')
column_names = df.columns

app = Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)
server = app.server

app.layout = html.Div([
    dbc.Row([
        dbc.Col(html.P('Please select the x axis:', style={'font-size': 20}), width={'size': 6}, lg=4),
        dbc.Col(dcc.Dropdown(
            id='xaxis-column',
            options=column_names,
            value='total_percipitation(in)',
        ), width={'size': 6, 'offset': 0}, lg=4)
    ]),
    dbc.Row([
        dbc.Col(html.P('Please select the y axis:', style={'font-size': 20}), width={'size': 6}, lg=4),
        dbc.Col(dcc.Dropdown(
            id='yaxis-column',
            options=column_names,
            value='Corn_Yield(BU/ACRE)',
        ), width={'size': 6, 'offset': 0}, lg=4)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='indicator-graphic'), width={'size': 6, 'offset': 2}, lg=6)
    ])
])


# app.layout = html.Div([
#     html.Div([
#
#         html.Div([
#             dcc.Dropdown(
#                 column_names,
#                 'total_percipitation(in)',
#                 id='xaxis-column'
#             )
#         ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
#
#         html.Div([
#             dcc.Dropdown(
#                 column_names,
#                 'Corn_Yield(BU/ACRE)',
#                 id='yaxis-column'
#             )
#         ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
#
#         dcc.Graph(id='indicator-graphic')
#     ]),
#
#
# ])

@app.callback(
    Output('indicator-graphic', 'figure'),
    Input('xaxis-column', 'value'),
    Input('yaxis-column', 'value'))
def update_graph(xaxis_column_name, yaxis_column_name):
    fig = px.scatter(x=df[xaxis_column_name],
                     y=df[yaxis_column_name])

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0})

    fig.update_xaxes(title=xaxis_column_name)

    fig.update_yaxes(title=yaxis_column_name)

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
