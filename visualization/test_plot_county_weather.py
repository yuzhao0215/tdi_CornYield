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
from plotly.subplots import make_subplots

import datetime

df = pd.read_csv("../data/17001_weather.csv", index_col=None)
corn_yields = pd.read_csv("../data/corn_yield_with_fips.csv", index_col=None)
corn_yields['CountyFIPS'] = corn_yields['CountyFIPS'].apply(lambda x: str(x).strip(".0")).copy()

corn_yields = corn_yields[corn_yields['CountyFIPS'] == '17001']

df["county_fips"] = df["county_fips"].apply(lambda x: str(x).strip(".0")).copy()
df["year"] = df["date"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").year).copy()

df = pd.merge(df, corn_yields, how="inner", left_on=["county_fips", "year"], right_on=["CountyFIPS", "Year"])

print(df.columns)

df['date'] = df['date'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d")).copy()

# fig = px.line(df, x='date', y="gdd")
fig = make_subplots(specs=[[{"secondary_y": True}]])

# fig.add_trace(
#     go.Scatter(x=df['date'], y=df['gdd'], mode="lines", name="gdd"), secondary_y=False
# )

# fig.add_trace(
#     go.Scatter(x=df['date'], y=df['pcpn'], mode="lines", name="pcpn"), secondary_y=False
# )

# fig.add_trace(
#     go.Scatter(x=df['date'], y=df['maxt'], mode="markers", name="maxt"), secondary_y=False
# )

fig.add_trace(
    go.Scatter(x=df['date'], y=df['avgt'], mode="markers", name="avgt"), secondary_y=False
)

# fig.add_trace(
#     go.Scatter(x=df['date'], y=df['mint'], mode="markers", name="mint"), secondary_y=False
# )

fig.add_trace(
    go.Scatter(x=df['date'], y=df['Yield'], mode="lines", name="Yield"), secondary_y=True
)

fig.show()
