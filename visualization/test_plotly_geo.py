import datetime
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

import sys

sys.path.insert(0, "../scraper")
sys.path.insert(0, "../model")

from scraper_weather_mrcc import read_county_weather
from scraper_weather_mrcc import read_county_weather_now
from test_final_model import *

from joblib import dump, load


from plotly.subplots import make_subplots


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

offset = 3

app.layout = html.Div([
    dbc.Row([
        dbc.Col(html.P('Please select year:', style={'font-size': 20}), width={'size': 3, 'offset': offset}, lg=3),
        dbc.Col(dcc.Dropdown(
            id='year',
            options=np.arange(oldest_year, latest_year+1),
            value=latest_year,
        ), width={'size': 3, 'offset': 0}, lg=3)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='corn-yield-graph'), width={'size': 6, 'offset': offset}, lg=6)
    ]),
    dbc.Row([
        dbc.Col(html.P('Please select a state:', style={'font-size': 20}), width={'size': 3, 'offset': offset}, lg=3),
        dbc.Col(dcc.Dropdown(
            id='state-dropdown',
            options=list(states_alpha2county.keys()),
            value='IL',
        ), width={'size': 3, 'offset': 0}, lg=3)
    ]),
    dbc.Row([
        dbc.Col(html.P('Please select a county:', style={'font-size': 20}), width={'size': 3, 'offset': offset}, lg=3),
        dbc.Col(dcc.Dropdown(
            id='county-dropdown',
            value='ADAMS'
        ), width={'size': 3, 'offset': 0}, lg=3)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='corn-yield-year'), width={'size': 6, 'offset': offset}, lg=6)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='daily-weather-now'), width={'size': 6, 'offset': offset}, lg=6)
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

    # print(future_yields)

    fig = px.scatter(filtered_df, x='Year', y='Yield')
    fig.add_trace(go.Scatter(x=future_years.ravel(), y=future_yields, mode='markers', name='predicted yields'))
    fig.add_trace(go.Scatter(x=dates.ravel(), y=reg.predict(dates), mode='lines', name='linear regression'))
    return fig


@app.callback(
    Output('daily-weather-now', 'figure'),
    Input('state-dropdown', 'value'),
    Input('county-dropdown', 'value')
)
def update_current_weather(state, county):
    filter1 = corn_yield_df['State'] == state
    filtered = corn_yield_df[filter1]
    filter2 = corn_yield_df['County'] == county
    filtered = filtered[filter2]

    county_fips = filtered["CountyFIPS"].min()

    df_daily = read_county_weather_now(county_fips)
    actual_month = find_maximum_actual_month(df_daily)
    grow_month = actual_to_growing_month(actual_month)

    now = datetime.datetime.now()

    dates = []
    predicted_yields = []

    for gm in range(1, grow_month + 1):
        am = reverse_growing_month(gm)

        if am >= 11:
            start_year = now.year - 1
        else:
            start_year = now.year

        d = datetime.datetime(start_year, am, 15)

        temp_df = preprocess_by_month(df_daily, gm)

        model = load(f'../model/cache/{am}.joblib')
        dates.append(d)
        predicted_yields.append(model.predict(temp_df)[0])

    print(dates, predicted_yields)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # fig.add_trace(
    #     go.Scatter(x=df_daily['date'], y=df_daily['avgt'], mode="lines", name="avgt"), secondary_y=False
    # )
    fig.add_trace(
        go.Scatter(x=df_daily['date'], y=df_daily['pcpn'], mode="lines", name="pcpn"), secondary_y=False
    )
    fig.add_trace(go.Scatter(x=dates, y=predicted_yields, mode='markers', name='predicted yields',
                             marker=dict(
                                 color='yellow',
                                 size=20,
                                 line=dict(
                                     color='green',
                                     width=4
                                 )
                             ),
                             ), secondary_y=True)

    fig.update_xaxes(title_text='Time')
    fig.update_yaxes(title_text='Precipitation (inches)', secondary_y=False)
    fig.update_yaxes(title_text="Corn Yield Prediction (BU/ACRE)", secondary_y=True)

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)

