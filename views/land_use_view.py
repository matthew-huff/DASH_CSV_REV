#
#   Matthew Huff
#   12/29/2020
#
#   View for Land Use and Land Ownership in Plotly Dash
#
#   Designed to be completely unstyled for copy/paste purposes into a styled page
#
#   Included in a Land Use and Land Ownership analysis:
#           -Basic location/scenario selectors
#           -Ability to select (from checkboxes) land use type and land ownership
#           -Graphs on LCOE and mean_cf depending on selections

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import dash_table
import time
import json
import dash_bootstrap_components as dbc
import random
from dash.dependencies import Input, Output

from flask_caching import Cache
import uuid


# Point to datafile
dataFile = "../../outfile4.csv"


df = pd.read_csv(dataFile)
locs = ['All', 'Northeast', 'Southeast', 'West', 'Great Lakes', 'Interior']

land_ownership_list = ['BLM', 'FS', 'FWS', 'BOR', 'NPS', 'BIA', 'DOD', 'VA',
                       'DOJ', 'NASA', 'DOE', 'DOL', 'USDA', 'TVA', 'HHS', 'DOT', 'DOC', 'MWAA', 'GSA']

land_use_display_list = ['Open Water', 'Perennial Snow', 'Developed, Open Space', 'Developed, Low Intensity', 'Developed, Medium Intensity', 'Developed, High Intensity', 'Barren Land', 'Deciduous Forest', 'Evergreen Forest',
                         'Mixed Forest',  'Shrub', 'Grassland',  'Pasture/Hay', 'Cultivated Crops', 'Woody Wetlands', 'Emergent Herbaceious Wetlands']

land_use_data_list = ['water', 'perennial_snow', 'open_developed', 'low_developed', 'medium_developed', 'high_developed', 'barren', 'deciduous_forest',
                      'evergreen_forest', 'mixed_forest',  'shrub', 'grassland',  'pasture', 'cultivated', 'woody_wetlands', 'herbaceous_wetlands']


scenarios = ['Open Access - Mid', 'Open Access - Low', 'Open Access - Current', 'Baseline - Mid',
             'Baseline - Current', 'Baseline - Low', 'Legacy - Current', 'Existing Social Acceptance - Mid',
             'Mid Social Acceptance - Mid', 'Mid Social Acceptance - Low', 'Low Social Acceptance - Mid',
             'Low Social Acceptance - Low', 'Radar Limits - Mid',
             'Smart Bat Curtailment - Mid', 'Blanket Bat Curtailment - Mid',
             'Fed Land Exclusion - Mid', 'Limited Access - Mid']

important_scenarios = ['Open Access - Mid', 'Open Access - Low', 'Open Access - Current', 'Baseline - Mid',
                       'Baseline - Current', 'Baseline - Low', 'Legacy - Current', 'Existing Social Acceptance - Mid', 'Limited Access - Mid']


attributes = list(df.columns)
for i in land_use_data_list:
    try:
        attributes.remove(i)
    except:
        print(i)
        exit(0)
# attributes.remove(land_ownership_list)
for i in land_ownership_list:
    try:
        attributes.remove(i)
    except:
        print(i)
        exit(0)
attributes.remove('Unnamed: 0')
attributes.remove('Unnamed: 0.1')
attributes.remove('longitude')
attributes.remove('latitude')
app = dash.Dash(__name__,   external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True)


cache = Cache(app.server, config={
    'CACHE_TYPE': 'redis',
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory',

    # Number below is number of allowed concurrent users - a final version should be much more than 2.
    'CACHE_THRESHOLD': 2
})


# Location and scenario selector
# location_selector, scenario_selector


def serve_layout():
    session_id = str(uuid.uuid4())

    return html.Div([
        html.H1(id='empty'),
        html.Div(session_id, id='session_id', style={'display': 'none'}),
        html.Div(id='selectors'),
        html.Div(id='land_use_mapbox_output'),
        html.Div(id='land_use_graph_output')
    ])


app.layout = serve_layout()


@app.callback(
    Output('selectors', 'children'),
    [
        Input('empty', 'value')
    ]
)
def displaySelectors(empty):
    locations_content = html.Div([
        html.H6("Location"),
        dcc.Dropdown(
            id='location_selector',
            options=[
                {'label': i, 'value': i} for i in locs
            ],
            value='Select'
        ),


        html.H6("Scenario"),
        dcc.Dropdown(
            id='scenario_selector',
            options=[
                {'label': i, 'value': i} for i in scenarios
            ],
            value='Select'
        ),

        html.H6('Attribute'),
        dcc.Dropdown(
            id='land_use_data_selector',
            options=[
                # for j in land_use_data_list
                {'label': i, 'value': i} for i in attributes
            ],
            value='Select'
        ),
        dcc.Checklist(
            id='land_ownership_list_checklist',
            options=[
                {'label': i, 'value': i} for i in land_ownership_list
            ],
            value=[]
        ),
        dcc.Checklist(
            id='land_use_list_checklist',
            options=[
                {'label': i, 'value': j} for i, j in zip(land_use_display_list, land_use_data_list)
            ],
            value=[]
        )
    ])
    return locations_content


# Filtering data by user session ID for fast access to multiple components using the same set of filtered data,
# as shown by Example 4 in https://dash.plotly.com/sharing-data-between-callbacks
def filter_data(session_id, location, scenario, land_ownership, land_use):
    @cache.memoize()
    def query_and_serialize_data(session_id, location, scenario, land_ownership, land_use):
        newDF = df.copy()

        # If land_use list is empty, use all.  Else, include only that which has been selected
        if(len(land_use) != 0):
            newDF = newDF[(df[land_use] > 0).all(1)]

        # If ownership list is empty, use all.  Else, include only that which has been selected
        if(len(land_ownership) != 0):
            newDF = newDF[(df[land_ownership] > 0).all(1)]

        # Repeat above for location
        if(location != 'Select'):
            newDF = newDF.loc[newDF['lbnl_region'] == location]

        # A scenario must be chosen.
        if(scenario != 'Select'):
            newDF = newDF.loc[newDF['scenario'] == scenario]
        else:
            return

        return newDF.to_json()

    return pd.read_json(query_and_serialize_data(session_id, location, scenario, land_ownership, land_use))


# Mapbox output that takes intermediate value from function and returns a display onto a map
@app.callback(
    Output('land_use_mapbox_output', 'children'),
    [
        Input('session_id', 'value'),
        Input('location_selector', 'value'),
        Input('scenario_selector', 'value'),
        Input('land_ownership_list_checklist', 'value'),
        Input('land_use_list_checklist', 'value'),
        Input('land_use_data_selector', 'value')
    ]
)
def update_land_use_map(session_id, location, scenario, land_ownership, land_use, attr):
    data = filter_data(session_id, location, scenario,
                       land_ownership, land_use)
    fig = px.scatter_mapbox(data, lat='latitude', lon='longitude',
                            size_max=3, color=attr, zoom=3, height=850)
    fig.update_layout(mapbox_style='open-street-map')
    return html.Div([
        html.Div([
            dcc.Graph(
                 figure=fig
                 )
        ])])


@app.callback(
    Output('land_use_graph_output', 'children'),
    [
        Input('session_id', 'value'),
        Input('location_selector', 'value'),
        Input('scenario_selector', 'value'),
        Input('land_ownership_list_checklist', 'value'),
        Input('land_use_list_checklist', 'value'),
    ]
)
def land_use_lcoe_cf_graph(session_id, location, scenario, land_ownership, land_use):
    data = filter_data(session_id, location, scenario,
                       land_ownership, land_use)
    fig = px.scatter(data, x="total_lcoe", y="mean_cf")
    return html.Div(
        [
            dcc.Graph(figure=fig)
        ],
        style={'padding-top': 40})


if __name__ == '__main__':
    app.run_server(debug=True,
                   dev_tools_ui=False,
                   dev_tools_props_check=False)
