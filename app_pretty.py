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
import app_pretty_mapbox
import random

diffDF = 0
devDF = 0
DIFF_FN = ""
currentSelectors = 0
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

currDF = 0


dataFile = "../outfile_extended.csv"
locs = ['All', 'Northeast', 'Southeast', 'West', 'Great Lakes', 'Interior']

scenarios = ['Open Access - Mid', 'Open Access - Low', 'Open Access - Current', 'Baseline - Mid',
             'Baseline - Current', 'Baseline - Low', 'Legacy - Current', 'Existing Social Acceptance - Mid',
             'Mid Social Acceptance - Mid', 'Mid Social Acceptance - Low', 'Low Social Acceptance - Mid',
             'Low Social Acceptance - Low', 'Radar Limits - Mid',
             'Smart Bat Curtailment - Mid', 'Blanket Bat Curtailment - Mid',
             'Fed Land Exclusion - Mid', 'Limited Access - Mid']

important_scenarios = ['Open Access - Mid', 'Open Access - Low', 'Open Access - Current', 'Baseline - Mid',
                       'Baseline - Current', 'Baseline - Low', 'Legacy - Current', 'Existing Social Acceptance - Mid', 'Limited Access - Mid']

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


currentArr = []
currentArrMax = 0
currentArrMin = 0

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
                                style={'padding-top': 10}),
                    html.Div(id='other_output_graphs',
                             style={'padding-top': 10})
                ],  width=3),



                dbc.Col([
                    html.Div(id='mapbox_output', style={'padding-top': 50})
                ], width=9)

            ]),

            # html.Div(id='output_graphs'),

        ])

    ], style={'padding': 40})





])


@ app.callback(
    dash.dependencies.Output('location_scenario_selector', 'children'),
    [dash.dependencies.Input('num_scenarios', 'value'),
     dash.dependencies.Input('view_selector', 'value')]
)
def locationSelector(numScenarios, view):
    locations_content = []
    if(view == 'Data'):
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

    elif(view == 'Deviations'):

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

        if(numScenarios == 2):
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

                    ]),
                    dbc.Row([
                        html.Div([
                            dbc.Button("Generate Diff", id='generate_btn',
                                       className='mr-1')
                        ], style={'padding-top': 20, 'padding-left': 20}),
                        html.Div(id='filename_text')

                    ])
                ])]

    return dbc.Card(locations_content, color='secondary', outline=True)


############
#
#       MAPBOX SELECTORS
#
#
############
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


@app.callback(
    dash.dependencies.Output('mapbox_g2_selector', 'children'),
    [dash.dependencies.Input('num_scenarios', 'value')]
)
def mapbox_g2_selector_func(num_scenarios):
    return dcc.Dropdown(
        id='mapbox_g2_selector_dropdown',
        options=[
            {'label': i, 'value': i} for i in attributes
        ],
        value='mean_cf'
    )
###########################################


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
                     y=np.flip(y_sorted), height=height, template='simple_white')
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


######################################################################
#
#                   MAPBOX_OUTPUT
#
######################################################################
@app.callback(
    dash.dependencies.Output('mapbox_output', 'children'),
    [dash.dependencies.Input('num_scenarios', 'value'),
     dash.dependencies.Input('view_selector', 'value')]
)
def mapbox_output_func(num_scenarios, view):
    content = []
    if(view == 'Data'):
        if(num_scenarios == 1):
            return html.Div(id='mapbox_area_1')
        elif(num_scenarios == 2):
            return html.Div(id='mapbox_area_2')
    elif(view == 'Deviations'):
        return html.Div(id='diff_card_output')


#######
#
#       MAPBOX AREA 1
#
#######

@app.callback(
    dash.dependencies.Output('mapbox_area_1', 'children'),
    [dash.dependencies.Input('num_scenarios', 'value')]
)
def mapbox_g1_func(num_scenarios):
    content = [
        dbc.CardHeader(
            html.Div(id='mapbox_g1_selector')
        ),
        dbc.CardBody([
            html.Div(id='output_mapbox_1'),

            html.Div(id="slider_output"),
            dcc.Interval(
                id='interval_strip',
                interval=100,
                n_intervals=0
            ),
            # html.Div(id="strip_interval_graph", style={'padding-top': 100}),
            html.Div(id='output_graphs')
        ])
    ]

    return html.Div([
        dbc.Card(content, color='secondary', outline=True),


    ])


@app.callback(
    dash.dependencies.Output('strip_interval_graph', 'children'),
    [
        dash.dependencies.Input('location_dropdown_g1', 'value'),
        dash.dependencies.Input('scenario_dropdown_g1', 'value'),
        dash.dependencies.Input('mapbox_g1_selector_dropdown', 'value'),
        dash.dependencies.Input('num_scenarios', 'value'),
        dash.dependencies.Input('interval_strip', 'n_intervals')
    ]
)
def update_strip(location, scenario, attribute, num_scenarios, n):
    # arr = df[scenario+":"+attribute].to_numpy().astype(float)
    indexVals = random.sample(range(0, len(currentArr)-1), 20)
    values = []

    for i in indexVals:
        values.append(currentArr[i])

    y = np.zeros(10)
    x = px.strip(y, values, orientation='h', range_x=[
                 currentArrMin, currentArrMax])
    return dcc.Graph(
        figure=x
    )


@ app.callback(
    dash.dependencies.Output('output_mapbox_1', 'children'),
    [dash.dependencies.Input('location_dropdown_g1', 'value'),
     dash.dependencies.Input('scenario_dropdown_g1', 'value'),
     dash.dependencies.Input('mapbox_g1_selector_dropdown', 'value'),
     dash.dependencies.Input('num_scenarios', 'value'),
     dash.dependencies.Input('slider', 'value')])
def update_map_g1(loc, scenario, dataVal, num_scenarios, slider_vals):
    global currentArr, currentArrMax, currentArrMin
    if(scenario == 'Select'):
        return
    size = 4
    zoom = 5
    if(loc == 'All'):
        newDF = df.copy()
    else:
        newDF = df.loc[df['Existing Social Acceptance - Mid:lbnl_region'] == loc]

    currentArr = newDF[scenario+":"+dataVal].to_numpy().astype(float)
    currentArrMax = np.max(currentArr)
    currentArrMin = np.min(currentArr)
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
        sortedDF = newDF.copy()
        sortedDF = sortedDF.sort_values(by=[scenario+':mean_lcoe'])

        sum_rows = 0
        min_counter = 0
        counter = 0
        # total = slider_vals[1]

        min_val = slider_vals[0]

        for row in sortedDF[scenario+':'+data]:
            if(sum_rows < min_val):
                sum_rows += row
                min_counter += 1
            else:
                break
        sum_rows = 0
        total = slider_vals[1]
        for row in sortedDF[scenario+':'+data]:
            if(sum_rows < total):
                sum_rows += row
                counter += 1
            else:
                break

        d = sortedDF[min_counter:counter]

        """
        size_arr = np.full(len(newDF[scenario+":"+data]), size)
        fig = px.scatter_mapbox(d, lat='latitude', lon='longitude', size=size_arr, hover_data=[
            scenario+":"+data], size_max=size, color=scenario+":"+data, zoom=zoom, height=height)
            """
        fig = px.scatter_mapbox(d, lat='latitude', lon='longitude',
                                color=scenario+":"+data, zoom=zoom, height=height)
        fig.update_layout(mapbox_style='open-street-map')
        # return(type(slider_vals))
        fig.update_layout(xaxis=dict(
            rangeslider=dict(
                visible=True
            )
        ))
        return dcc.Graph(
            figure=fig
        )


########
#
#       MAPBOX AREA 2
#
########

@app.callback(
    dash.dependencies.Output('mapbox_area_2', 'children'),
    [dash.dependencies.Input('num_scenarios', 'value')]
)
def mapbox_2_func(num_scenarios):
    content = [
        dbc.CardHeader(

            dbc.Row([
                dbc.Col(html.Div(id='mapbox_g1_selector'), width=5),
                dbc.Col(html.Div(id='mapbox_g2_selector'), width=5)
            ], justify='around')
        ),
        dbc.CardBody([

            dbc.Row([
                dbc.Col(html.Div(id='output_mapbox_1'), width=6),
                dbc.Col(html.Div(id='output_mapbox_2'), width=6)
            ], justify='around')

        ])
    ]

    return dbc.Card(content, color='secondary', outline=True)

##########
#
#       Slider for mapbox
#
##########


@app.callback(
    dash.dependencies.Output('slider_output', 'children'),
    [
        dash.dependencies.Input('location_dropdown_g1', 'value'),
        dash.dependencies.Input('scenario_dropdown_g1', 'value'),
        dash.dependencies.Input('mapbox_g1_selector_dropdown', 'value'),
        dash.dependencies.Input('num_scenarios', 'value'),
    ]
)
def slider_output(location, scenario, attribute, num_scenarios):
    if(scenario == 'Select'):
        return
    if(location == 'Select'):
        newDF = df.copy()
    else:
        newDF = df.loc[df['Existing Social Acceptance - Mid:lbnl_region'] == location]

    arr = newDF[scenario + ":" + attribute].to_numpy().astype(float)

    total = int(np.sum(arr))

    first_25_mark = int(total * 0.25)
    first_50_mark = int(total * .5)
    first_75_mark = int(total * .75)
    return html.Div([
        dcc.RangeSlider(
            id='slider',
            min=0,
            max=total,
            step=1,
            value=[0, total],
            marks={
                first_25_mark: {'label': '{}'.format(first_25_mark)},
                first_50_mark: {'label': '{}'.format(first_50_mark)},
                first_75_mark: {'label': '{}'.format(first_75_mark)}
            },
            tooltip={'placement': 'bottom'}
        )
    ])


@ app.callback(
    dash.dependencies.Output('output_mapbox_2', 'children'),
    [dash.dependencies.Input('location_dropdown_g2', 'value'),
     dash.dependencies.Input('scenario_dropdown_g2', 'value'),
     dash.dependencies.Input('mapbox_g2_selector_dropdown', 'value'),
     dash.dependencies.Input('num_scenarios', 'value'), ])
def update_map_g2(loc, scenario, dataVal, num_scenarios):
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
        fig.update_traces(marker_colorbar_title_text="",
                          selector=dict(type='scattermapbox'))
        fig.update_layout(mapbox_style='open-street-map',
                          legend_orientation="h")
        return dcc.Graph(
            figure=fig
        )

##########
#
#       Deviations
#           Outputting to diff_mapbox_output
##########


@ app.callback(
    dash.dependencies.Output('diff_card_output', 'children'),
    [
        dash.dependencies.Input('num_scenarios', 'value')
    ]
)
def output_diff(val):
    content = []
    if(val == 1):
        content = [
            # Card header will hold dropdown selector
            dbc.CardHeader(html.Div(id='dev_dropdown')),
            dbc.CardBody(
                html.Div([
                    html.Div(id='dev_selector'),
                    html.Div(id='dev_mapbox_output')
                ]))
        ]
    elif(val == 2):
        content = [
            dbc.CardHeader(
                html.Div([
                    dcc.Dropdown(id='diff_selector',
                                 options=[
                                    {'label': col.split(':')[1], 'value':col.split(':')[1]} for col in cols
                                 ],
                                 value='Select value...'),
                ], className="six columns"),
            ),
            dbc.CardBody(html.Div(id='diff_mapbox_output'))
        ]
    return dbc.Card(content, color='secondary', outline=True)
##########
#
#       Boxplots, Histograms, etc
#
##########


@ app.callback(
    dash.dependencies.Output('output_graphs', 'children'),
    [dash.dependencies.Input('num_scenarios', 'value')]
)
def return_output_graphs(num_scenarios):

    if(num_scenarios == 1):
        return html.Div(id='output_graph_1')
    elif(num_scenarios == 2):
        return html.Div(id='output_graphs_2')


# Output Graphs for 1 scenario
@app.callback(
    dash.dependencies.Output('output_graph_1', 'children'),
    [dash.dependencies.Input('mapbox_g1_selector_dropdown', 'value'),
     dash.dependencies.Input('scenario_dropdown_g1', 'value'),
     dash.dependencies.Input('location_dropdown_g1', 'value')]
)
def outputGraph_1(attr, scen, loc):

    if(loc == 'All'):
        newDF = df.copy()
    else:
        fig = getViolinPlot(loc, scen, attr)

    return dcc.Graph(
        figure=fig
    )


# Interval update of strip


def getViolinPlot(location, scenario, attribute):
    if(location == 'All'):
        newDF = df.copy()
    else:
        newDF = df.loc[df['Existing Social Acceptance - Mid:lbnl_region'] == location]
    val = scenario+":"+attribute
    fig = px.violin(newDF, x=newDF[val], box=True,
                    template='simple_white')
    # fig = go.Figure(data=go.hea)
    fig.update_layout(xaxis_zeroline=False)
    return fig


def getHistogramPlot(location, scenario, attribute):
    newDF = df.loc[df['Existing Social Acceptance - Mid:lbnl_region'] == location]
    val = scenario+":"+attribute
    fig = px.histogram(newDF, x=val)
    return fig


def getCumSumPlot(location, scenario, attribute):
    newDF = df.loc[df['Existing Social Acceptance - Mid:lbnl_region'] == location]
    data = newDF[scenario+":"+attribute].to_numpy(dtype=float)
    cs = np.cumsum(data)

    fig = px.scatter(x=np.arange(len(cs)),
                     y=cs)

    return fig


####################
#
#   DIFFS
#
####################


@ app.callback(
    dash.dependencies.Output('diff_mapbox_output', 'children'),
    [dash.dependencies.Input('diff_selector', 'value')],
    [dash.dependencies.State('filename_text', 'children'),
     dash.dependencies.State('location_dropdown_g1', 'value')]
)
def view_diff(diff_var, fn, loc):
    if(fn == ""):
        return html.H5("Generate a diff between two scenarios")
    if(type(fn) != None):
        try:
            newDF = pd.read_csv(fn)
        except ValueError:
            return html.H5("Generate a diff between two scenarios")

        size = 5
        zoom = 5
        if(loc == 'All'):
            size = 2
            zoom = 3

        size_arr = np.full(len(newDF[diff_var]), size)
        fig = px.scatter_mapbox(newDF, lat='latitude', lon='longitude', hover_data=[
            diff_var], size=size_arr, size_max=size, color=diff_var, zoom=zoom, height=1200)
        fig.update_layout(mapbox_style='open-street-map')
        return dcc.Graph(
            figure=fig
        )


@ app.callback(
    dash.dependencies.Output('filename_text', 'children'),
    [
        dash.dependencies.Input('generate_btn', 'n_clicks'),
    ],
    state=[
        dash.dependencies.State('location_dropdown_g1', 'value'),
        dash.dependencies.State('location_dropdown_g2', 'value'),
        dash.dependencies.State('scenario_dropdown_g1', 'value'),
        dash.dependencies.State('scenario_dropdown_g2', 'value'),
        # dash.dependencies.State('diff_selector', 'value')
    ]
)
def generate_diff(n_clicks, loc1, loc2, scen1, scen2):
    DIFF_FN = loc1+'_'+scen1+scen2 + ".csv"
    if(scen1 == 'All' or scen2 == 'All'):
        return html.H5("Neither scenario can be All")
    if(scen1 == scen2):
        return html.H5("Scenarios cannot be the same")
    if(loc1 != loc2):
        return html.H5("Locations must be the same")

    variables = [col.split(':')[1] for col in cols]

    newDF1 = df.loc[df['Existing Social Acceptance - Mid:lbnl_region'] == loc1]
    lats = newDF1['latitude']
    lons = newDF1['longitude']
    newDF = {'latitude': lats, 'longitude': lons}

    for column in list(df.columns):
        if(column.startswith(scen1) or column.startswith(scen2)):
            newDF[column] = df[column]

    newDF = pd.DataFrame.from_dict(newDF, dtype='str')
    diff_dict = {'latitude': newDF['latitude'],
                 'longitude': newDF['longitude']}

    for var in variables:
        try:
            scen1_data = newDF[scen1+":"+var].astype(float)
            scen2_data = newDF[scen2+":"+var].astype(float)
        except ValueError:
            continue

        try:
            diff_data = np.subtract(scen1_data.to_numpy().astype(
                float), scen2_data.to_numpy().astype(float))
            diff_dict[var] = diff_data
        except TypeError:
            pass

    diffDF = pd.DataFrame.from_dict(diff_dict, dtype='str')

    diffDF.to_csv(DIFF_FN)
    diffDF = pd.read_csv(DIFF_FN)
    return DIFF_FN

####################################
#
#   Deviation functions (std, rsd, quartiles) for single scenario viewing
#
####################################


@ app.callback(
    dash.dependencies.Output('dev_selector', 'children'),
    [dash.dependencies.Input('view_selector', 'value')]
)
def generate_deviations_options(val):

    devAttrs = []
    for attr in attributes:
        devAttrs.append(attr + '_std')
        devAttrs.append(attr + '_rsd')
        devAttrs.append(attr + '_minimum')
        devAttrs.append(attr + '_quartile1')
        devAttrs.append(attr + '_quartile2')
        devAttrs.append(attr + '_quartile3')
        devAttrs.append(attr + '_quartile4')

    return html.Div([

        html.Div([
            dcc.Checklist(
                id='dev_selector_checklist',
                options=[
                    {'label': i, 'value': i} for i in important_scenarios
                ],
                value=[i for i in important_scenarios],

            )
        ], className="eight columns"),



    ], className="row")


def getSTD(arrs, n):
    arr = np.vstack(arrs)
    if(n):
        v = np.nanstd(arr, axis=0)
    else:
        v = np.std(arr, axis=0)
    return v


def getQuartiles(arrs, n):
    arr = np.vstack(arrs)
    if(n):
        minimum = np.nanmin(arr, axis=0)
        q1 = np.nanquantile(arr, 0.25, axis=0)
        q2 = np.nanquantile(arr, 0.5, axis=0)
        q3 = np.nanquantile(arr, 0.75, axis=0)
        q4 = np.nanquantile(arr, 1, axis=0)
    else:
        minimum = np.min(arr, axis=0)
        q1 = np.quantile(arr, 0.25, axis=0)
        q2 = np.quantile(arr, 0.5, axis=0)
        q3 = np.quantile(arr, 0.75, axis=0)
        q4 = np.quantile(arr, 1, axis=0)
    return minimum, q1, q2, q3, q4


def getNumZeros(arrs, n):
    arr = np.vstack(arrs)
    isna = np.count_nonzero(np.isnan(arr), axis=0)
    for i, j in enumerate(isna):
        isna[i] = len(arrs)-j
    return isna


def getMean(arrs, attr, n):
    # print("Attribute: " + attr + ", Len of arrs = " + str(len(arrs)))
    arr = np.vstack(arrs)
    if(n):
        a = np.nanmean(arr, axis=0)
    else:
        a = np.mean(arr, axis=0)

    # s = time.time()
    v = getSTD(arrs, n)
    # print("Time getSTD: " + str(s - time.time()))
    rsd = (100 * v) / a
    return a, v, rsd


def generate_deviations_DF(selectors, dropdown):
    global devDF
    newDF = df.copy()

    varList = []
    avgDict = {}

    for attr in attributes:
        for column in list(newDF.columns):
            try:
                if(column.split(':')[1] == attr and column.split(':')[0] in selectors):
                    varList.append(newDF[column].astype(float).to_numpy())
            except IndexError:
                avgDict[column] = newDF[column].to_numpy()
            except ValueError:
                pass

        if(len(varList) == 0):
            pass
        else:
            # s = time.time()
            avg, std, rsd = getMean(varList, attr, 1)
            # print("Time getMean: " + str(s - time.time()))

            # s = time.time()
            numScenarios = getNumZeros(varList, 1)
            # print("Time numZeros: " + str(s - time.time()))

            # s = time.time()
            minimum, q1, q2, q3, q4 = getQuartiles(varList, 0)
            # print("Time getQuartiles: " + str(s - time.time()))

            avgDict[attr] = avg
            avgDict[attr+"_std"] = std
            avgDict[attr+"_rsd"] = rsd
            avgDict[attr+"_minimum"] = minimum
            avgDict[attr+"_quartile1"] = q1
            avgDict[attr+"_quartile2"] = q2
            avgDict[attr+"_quartile3"] = q3
            avgDict[attr+"_quartile4"] = q4
            varList = []
    avgDict["num_scenarios"] = numScenarios
    avgDict['lbnl_region'] = newDF['Existing Social Acceptance - Mid:lbnl_region']
    devDF = pd.DataFrame.from_dict(avgDict)


@app.callback(
    dash.dependencies.Output('dev_dropdown', 'children'),
    [dash.dependencies.Input('view_selector', 'value')]
)
def generate_dev_dropdown(val):
    devAttrs = []
    for attr in attributes:
        devAttrs.append(attr + '_std')
        devAttrs.append(attr + '_rsd')
        devAttrs.append(attr + '_minimum')
        devAttrs.append(attr + '_quartile1')
        devAttrs.append(attr + '_quartile2')
        devAttrs.append(attr + '_quartile3')
        devAttrs.append(attr + '_quartile4')

    return html.Div([
        dcc.Dropdown(

                    id='dev_dropdown_options',
                    options=[
                        {'label': da, 'value': da} for da in devAttrs
                    ],
                    value='mean_cf_std'
                    ),

    ])


@app.callback(
    dash.dependencies.Output('deviations_view', 'children'),
    [dash.dependencies.Input('view_selector', 'value')]
)
def gen_deviations_view(val):
    devAttrs_std = []
    devAttrs_q = []
    for attr in attributes:
        devAttrs_std.append(attr + '_std')
        devAttrs_std.append(attr + '_rsd')
        devAttrs_q.append(attr + '_minimum')
        devAttrs_q.append(attr + '_quartile1')
        devAttrs_q.append(attr + '_quartile2')
        devAttrs_q.append(attr + '_quartile3')
        devAttrs_q.append(attr + '_quartile4')

    return html.Div([
        html.Div([
            dcc.Dropdown(
                id='dev_dropdown_std',
                options=[
                    {'label': i, 'value': i} for i in devAttrs_std
                ],
                value="mean_cf_std"
            ),
            html.Div(id="dev_data_std")
        ],
            className="six columns"
        ),
        html.Div([
            dcc.Dropdown(
                id="dev_dropdown_q",
                options=[
                    {'label': i, 'value': i} for i in devAttrs_q
                ],
                value="mean_cf_quartile1"
            ),

            html.Div(id="dev_data_q")
        ],
            className="six columns"
        )
    ])


@app.callback(
    dash.dependencies.Output('dev_mapbox_output', 'children'),
    [
        dash.dependencies.Input('dev_selector_checklist', 'value'),
        dash.dependencies.Input('dev_dropdown_options', 'value'),
        dash.dependencies.Input('location_dropdown_g1', 'value')
    ]
)
def generate_deviations_output(checklist, dropdown, location):
    global currentSelectors
    global devDF

    if(checklist != currentSelectors):
        currentSelectors = checklist
        generate_deviations_DF(checklist, dropdown)
    #size_arr = devDF['num_scenarios'].to_numpy()

    # If location isn't empty, make a local copy with only those locations satisfied
    if(location == 'Select' or location == 'All'):
        localDF = devDF.copy()
    else:
        localDF = devDF.loc[devDF['lbnl_region'] == location]
        # return html.H5(len(localDF['lbnl_region']))
    size_arr = localDF['num_scenarios'].to_numpy()
    fig = px.scatter_mapbox(localDF, lat='latitude', lon='longitude', size=size_arr,
                            size_max=3, color=dropdown, zoom=3, height=850)
    fig.update_layout(mapbox_style='open-street-map')

    return html.Div([
        html.Div([
            dcc.Graph(
                 figure=fig
                 )
        ])])


@app.callback(
    dash.dependencies.Output('other_output_graphs', 'children'),
    [
        dash.dependencies.Input('num_scenarios', 'value'),
        dash.dependencies.Input('view', 'value')
    ]
)
def generate_other_graphs(num_scenarios, view):
    if(view == 'Data'):
        return
    elif(view == 'Deviations'):
        if(num_scenarios == 1):
            return html.Div('deviations_1')


"""
@app.callback(
    dash.dependencies.Output('deviations_1'),
    [
        dash.dependencies.Input('dev_selector_checklist', 'value'),
        dash.dependencies.Input('dev_dropdown_options', 'value')
    ]
)
# def deviations_1_output(selectors, dropdown):
# Histogram plot of dropdown
"""
if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_ui=False,
                   dev_tools_props_check=False)
