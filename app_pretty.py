import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import dash_table
import time
import dash_bootstrap_components as dbc

diffDF = 0
devDF = 0
DIFF_FN = ""
currentSelectors = 0
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


dataFile = "../outfile_extended.csv"
locs = ['All', 'Northeast', 'Southeast', 'West', 'Great Lakes', 'Interior']

scenarios = ['Open Access - Mid', 'Open Access - Low', 'Open Access - Current', 'Baseline - Mid',
             'Baseline - Current', 'Baseline - Low', 'Legacy - Current', 'Existing Social Acceptance - Mid',
             'Mid Social Acceptance - Mid', 'Mid Social Acceptance - Low', 'Low Social Acceptance - Mid',
             'Low Social Acceptance - Low', 'Radar Limits - Mid',
             'Smart Bat Curtailment - Mid', 'Blanket Bat Curtailment - Mid',
             'Fed Land Exclusion - Mid', 'Limited Access - Mid']

df = pd.read_csv(dataFile)
colList = list(df.columns)
x = []
for col in colList:
    if(col.startswith('Existing Social Acceptance')):
        x.append(col)
attributes = [i.split(':')[1] for i in x]

cols = list(df.columns)
cols.remove('latitude')
cols.remove('longitude')
newCols = []
for col in cols:
    if(col.startswith('Existing Social Acceptance')):
        newCols.append(col)
cols = newCols


# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__,   external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True)


options_content = [
    dbc.CardHeader("Options Panel"),
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.H6("Scenarios"),
                dcc.Dropdown(
                    id='num_scenarios',
                    options=[
                        {'label': '1', 'value': 1},
                        {'label': '2', 'value': 2}
                    ],
                    value=1
                ),
            ]),

            dbc.Col([

                html.H6(children="LCOE Parameter"),
                dcc.Dropdown(
                    id='lcoe_selection',
                    options=[
                        {'label': 'Mean LCOE', 'value': 'mean_lcoe'},
                        {'label': 'Total LCOE', 'value': 'total_lcoe'}
                    ],
                    value='total_lcoe'
                ),
            ])
        ]),

        dbc.Row([
            dbc.Col([
                html.H6("View"),
                dcc.Dropdown(
                    id='view_selector',
                    options=[
                        {'label': 'Data', 'value': 'Data'},
                        {'label': 'Deviations', 'value': 'Deviations'}
                    ],
                    value='Data'
                )
            ])
        ])

    ])
]

locations_content = [
    dbc.CardHeader("Location Selector"),
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                 html.H6("Location"),
                 dcc.Dropdown(
                     id='location_selector',
                     options=[
                         {'label': i, 'value': i} for i in locs
                     ],
                     value='Select'
                 )
                 ])
        ]),
        dbc.Row([
            dbc.Col([
                html.H6("Scenario"),
                dcc.Dropdown(
                    id='scenario_selector',
                    options=[
                        {'label': i, 'value': i} for i in scenarios
                    ],
                    value='Select'
                )
            ])
        ])
    ])
]


app.layout = html.Div([

    # Upper panel
    html.Div([

        html.Div([

            html.Div([
                html.H4("Wind Turbine Dashboard"),
            ], style={'textAlign': 'center'}),


            dbc.Row([

                dbc.Col([
                    html.Div(dbc.Card(options_content,
                                      color='secondary', outline=True), style={'padding-top': 50}),

                    html.Div(id='location_scenario_selector',
                                style={'padding-top': 10}),
                    html.Div(id='output_curve_g1',
                                style={'padding-top': 10})
                ],  width=4),



                dbc.Col([
                    html.Div(id='mapbox_g1', style={'padding-top': 50})
                ], width=8)

            ]),
        ])

    ], style={'padding': 40})





])

"""
            dbc.Row([
                dbc.Col(dbc.Card(options_content,
                                 color='secondary', outline=True), width=4)
            ]),

            html.Div([
                dbc.Row([
                    dbc.Col(html.Div(id='location_scenario_selector'), width=4)
                ])
            ], style={'padding-top': 10})

        ], className="mb-4")



    ], style={'padding': 40})

    # Data panel
"""


@ app.callback(
    dash.dependencies.Output('location_scenario_selector', 'children'),
    [dash.dependencies.Input('num_scenarios', 'value')]
)
def locationSelector(numScenarios):
    locations_content = []
    if(numScenarios == 1):

        locations_content = [
            dbc.CardHeader("Location and Scenarios"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("Location"),
                        dcc.Dropdown(
                            id='location_dropdown_g1',
                            options=[
                                {'label': i, 'value': i} for i in locs
                            ],
                            value='Select'
                        )
                    ])
                ]),
                dbc.Row([
                    dbc.Col([
                        html.H6("Scenario"),
                        dcc.Dropdown(
                            id='scenario_dropdown_g1',
                            options=[
                                {'label': i, 'value': i} for i in scenarios
                            ],
                            value='Select'
                        )
                    ])
                ])
            ])
        ]

    elif(numScenarios == 2):
        locations_content = [
            dbc.CardHeader("Location and Scenarios"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("Location 1"),
                        dcc.Dropdown(
                            id='location_dropdown_g1',
                            options=[
                                {'label': i, 'value': i} for i in locs
                            ],
                            value='Select'
                        )
                    ]),


                    dbc.Col([
                        html.H6("Scenario 1"),
                        dcc.Dropdown(
                            id='scenario_dropdown_g1',
                            options=[
                                {'label': i, 'value': i} for i in scenarios
                            ],
                            value='Select'
                        )
                    ])

                ]),

                dbc.Row([
                    dbc.Col([
                        html.H6("Location 2"),
                        dcc.Dropdown(
                            id='location_dropdown_g2',
                            options=[
                                {'label': i, 'value': i} for i in locs
                            ],
                            value='Select'
                        )
                    ]),


                    dbc.Col([
                        html.H6("Scenario 2"),
                        dcc.Dropdown(
                            id='scenario_dropdown_g2',
                            options=[
                                {'label': i, 'value': i} for i in scenarios
                            ],
                            value='Select'
                        )
                    ])

                ])
            ])]

    return dbc.Card(locations_content, color='secondary', outline=True)


@app.callback(
    dash.dependencies.Output('mapbox_g1', 'children'),
    [dash.dependencies.Input('num_scenarios', 'value')]
)
def mapbox_g1_func(num_scenarios):
    content = [
        dbc.CardHeader(
            html.Div(id='mapbox_g1_selector')
        ),
        dbc.CardBody([
            html.Div(id='output_mapbox_g1')
        ])
    ]

    return dbc.Card(content, color='secondary', outline=True)


@app.callback(
    dash.dependencies.Output('mapbox_g1_selector', 'children'),
    [dash.dependencies.Input('num_scenarios', 'value')]
)
def mapbox_g1_selector_func(num_scenarios):
    return dcc.Dropdown(
        id='mapbox_g1_selector_dropdown',
        options=[
            {'label': i, 'value': i} for i in attributes
        ],
        value='mean_cf'
    )


@ app.callback(
    dash.dependencies.Output('output_curve_g1', 'children'),
    [dash.dependencies.Input('num_scenarios', 'value'),
     dash.dependencies.Input('location_dropdown_g1', 'value'),
     dash.dependencies.Input('scenario_dropdown_g1', 'value'),
     dash.dependencies.Input('lcoe_selection', 'value')])
def update_g1(num, loc, scenario, lcoe):
    if(scenario == 'Select'):
        return
    newDF = df.loc[df['Existing Social Acceptance - Mid:lbnl_region'] == loc]
    colList = list(newDF.columns)
    if(num == 1):
        height = 300
    elif(num == 2):
        height = 300
    if(scenario == 'All'):
        return
    else:
        for col in colList:
            if(col != 'latitude' or col != 'longitude' or (not col.startswith(scenario))):
                colList.remove(col)
        newDF.drop(colList, axis=1)

    y_sorted = np.sort(newDF[scenario + ':' + lcoe].to_numpy().astype(float))
    fig = px.scatter(x=np.arange(len(y_sorted)),
                     y=np.flip(y_sorted), height=height)
    sum_y = np.nansum(y_sorted)
    return html.Div([
        dcc.Graph(
            id='g1',
            figure=fig
        ),
        html.Div([
            html.H5("Value under LCOE curve: " + format(int(sum_y), ','))
        ], style={'textAlign': 'left'})
    ])


@ app.callback(
    dash.dependencies.Output('output_mapbox_g1', 'children'),
    [dash.dependencies.Input('location_dropdown_g1', 'value'),
     dash.dependencies.Input('scenario_dropdown_g1', 'value'),
     dash.dependencies.Input('mapbox_g1_selector_dropdown', 'value'),
     dash.dependencies.Input('num_scenarios', 'value'), ])
def update_map_g1(loc, scenario, dataVal, num_scenarios):
    if(scenario == 'Select'):
        return
    size = 4
    zoom = 5
    if(loc == 'All'):
        newDF = df.copy()
    else:
        newDF = df.loc[df['Existing Social Acceptance - Mid:lbnl_region'] == loc]
    data = dataVal
    if(num_scenarios == 1):
        height = 850
    elif(num_scenarios == 2):
        height = 600
    if(scenario == 'Select'):
        pass
    else:
        if(loc == 'All'):
            size = 2
            zoom = 3

        size_arr = np.full(len(newDF[scenario+":"+data]), size)
        fig = px.scatter_mapbox(newDF, lat='latitude', lon='longitude', size=size_arr, hover_data=[
            scenario+":"+data], size_max=size, color=scenario+":"+data, zoom=zoom, height=height)
        fig.update_layout(mapbox_style='open-street-map')
        return dcc.Graph(
            figure=fig
        )


if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_ui=False,
                   dev_tools_props_check=False)
