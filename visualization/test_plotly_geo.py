from urllib.request import urlopen
import json
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import numpy as np
import plotly.graph_objects as go


# open county geojson file for corn yield plotting on map
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

# read corn yield dataframe
corn_yield_df = pd.read_csv('../data/corn_yield.csv')

# fips file of counties
FipsDF = pd.read_csv('../data/fips2county.tsv', sep='\t', header='infer', dtype=str, encoding='latin-1')

# select columns for merging later
StateCountyDF = FipsDF[["CountyFIPS", "CountyName", "StateName", "StateAbbr"]].copy()

# convert county names to upper case
StateCountyDF['CountyName'] = StateCountyDF['CountyName'].str.upper().copy()

# add fips codes of each county
corn_yield_df = corn_yield_df.merge(StateCountyDF, how='left', left_on=['State', 'County'],
                                    right_on=["StateAbbr", "CountyName"])

# drop duplicate column
corn_yield_df.drop(columns=['State'])

# save corn_yield_df after merging
corn_yield_df.to_csv("../data/corn_yield_with_fips.csv", index=False)

# latest year of corn yield data
latest_year = corn_yield_df['Year'].max()
oldest_year = corn_yield_df['Year'].min()

# read dictionary contains states and counties
states_alpha2county = pickle.load(open("../scraper/county_names_dic.pkl", "rb"))

app = Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)
server = app.server

app.layout = html.Div([
    dbc.Row([
        dbc.Col(html.P('Please select year:', style={'font-size': 20}), width={'size': 3}, lg=3),
        dbc.Col(dcc.Dropdown(
            id='year',
            options=np.arange(oldest_year, latest_year+1),
            value=latest_year,
        ), width={'size': 3, 'offset': 0}, lg=3)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='corn-yield-graph'), width={'size': 6, 'offset': 0}, lg=6)
    ]),
    dbc.Row([
        dbc.Col(html.P('Please select a state:', style={'font-size': 20}), width={'size': 3}, lg=3),
        dbc.Col(dcc.Dropdown(
            id='state-dropdown',
            options=list(states_alpha2county.keys()),
            value='IL',
        ), width={'size': 3, 'offset': 0}, lg=3)
    ]),
    dbc.Row([
        dbc.Col(html.P('Please select a county:', style={'font-size': 20}), width={'size': 3}, lg=3),
        dbc.Col(dcc.Dropdown(
            id='county-dropdown',
            value='ADAMS'
        ), width={'size': 3, 'offset': 0}, lg=3)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='corn-yield-year'), width={'size': 6, 'offset': 0}, lg=6)
    ]),
])


@app.callback(
    Output('corn-yield-graph', 'figure'),
    Input('year', 'value'))
def update_map_graph(year):
    filtered_corn_yield_by_year = corn_yield_df[corn_yield_df['Year'] == year]
    min_yield = filtered_corn_yield_by_year['Yield'].min()
    max_yield = filtered_corn_yield_by_year['Yield'].max()
    fig = px.choropleth(filtered_corn_yield_by_year, geojson=counties, locations='CountyFIPS', color='Yield',
                        color_continuous_scale="Viridis",
                        range_color=(min_yield, max_yield),
                        scope="usa",
                        hover_data=["StateName", "CountyName", "Yield"]
                        )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return fig


@app.callback(
    Output('county-dropdown', 'options'),
    Input('state-dropdown', 'value')
)
def update_county_dropdown_options(selected_state):
    return [{'label': i, 'value': i} for i in states_alpha2county[selected_state]]


@app.callback(
    Output('corn-yield-year', 'figure'),
    Input('state-dropdown', 'value'),
    Input('county-dropdown', 'value')
)
def update_yield_year_graph(state, county):
    filter1 = corn_yield_df['StateAbbr'] == state

    filtered_df = corn_yield_df[filter1]

    filtered_df = filtered_df[filtered_df['CountyName'] == county]

    dates = filtered_df['Year'].values.reshape(-1, 1)
    corn_yields = filtered_df['Yield']

    reg = LinearRegression().fit(dates, corn_yields)
    future_years = np.arange(2022, 2025).reshape(-1, 1)
    future_yields = reg.predict(future_years)

    print(future_yields)

    fig = px.scatter(filtered_df, x='Year', y='Yield')
    fig.add_trace(go.Scatter(x=future_years.ravel(), y=future_yields, mode='markers', name='predicted yields'))
    fig.add_trace(go.Scatter(x=dates.ravel(), y=reg.predict(dates), mode='lines', name='linear regression'))
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)

