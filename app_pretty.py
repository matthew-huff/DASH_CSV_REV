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
                ],  width=3),



                dbc.Col([
                    html.Div(id='mapbox_output', style={'padding-top': 50})
                ], width=9)

            ]),

            html.Div(id='output_graphs')
        ])

    ], style={'padding': 40})





])


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




######################################################################
#
#                   MAPBOX_OUTPUT
#
######################################################################
@app.callback(
    dash.dependencies.Output('mapbox_output', 'children'),
    [dash.dependencies.Input('num_scenarios', 'value')]
)
def mapbox_output_func(num_scenarios):
    content = []
    if(num_scenarios == 1):
        
        return html.Div(id='mapbox_area_1')
    elif(num_scenarios==2):
        return html.Div(id='mapbox_area_2')



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
            html.Div(id="slider_text_boxes", style={'padding-top' : 100}),
        ])
    ]

    return html.Div([
        dbc.Card(content, color='secondary', outline=True)
    ])
    


@ app.callback(
    dash.dependencies.Output('output_mapbox_1', 'children'),
    [dash.dependencies.Input('location_dropdown_g1', 'value'),
     dash.dependencies.Input('scenario_dropdown_g1', 'value'),
     dash.dependencies.Input('mapbox_g1_selector_dropdown', 'value'),
     dash.dependencies.Input('num_scenarios', 'value'),
     dash.dependencies.Input('slider', 'value') ])
def update_map_g1(loc, scenario, dataVal, num_scenarios, slider_vals):
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
        sortedDF = newDF.copy()
        sortedDF = sortedDF.sort_values(by=[scenario+':mean_lcoe'])

        sum_rows=0
        counter = 0
        #total = slider_vals[1]

        total = slider_vals
        for row in sortedDF[scenario+':'+data]:
            if(sum_rows < total):
                sum_rows += row
                counter += 1
            else:
                break

        d = sortedDF[:counter]

        """
        size_arr = np.full(len(newDF[scenario+":"+data]), size)
        fig = px.scatter_mapbox(d, lat='latitude', lon='longitude', size=size_arr, hover_data=[
            scenario+":"+data], size_max=size, color=scenario+":"+data, zoom=zoom, height=height)
            """
        fig = px.scatter_mapbox(d, lat='latitude', lon='longitude', color=scenario+":"+data, zoom=zoom, height=height)
        fig.update_layout(mapbox_style='open-street-map')
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
    dash.dependencies.Output("slider_text_boxes", 'children'),
    [
        dash.dependencies.Input('location_dropdown_g1', 'value'),
        dash.dependencies.Input('scenario_dropdown_g1', 'value'),
        dash.dependencies.Input('mapbox_g1_selector_dropdown', 'value'),
        dash.dependencies.Input('num_scenarios', 'value'),
    ]
)
def slider_textbox(location,scenario,attribute,num):
    if(scenario=='Select'):
        return
    if(location == 'Select'):
        newDF = df.copy()

    return html.Div([
        dcc.Input(id="slider_low", type="text", placeholder="Min"),
        dcc.Input(id="slider_high", type="text", placeholder="Max")
    ])




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
    if(scenario=='Select'):
        return
    if(location == 'Select'):
        newDF = df.copy()
    else:
        newDF = df.loc[df['Existing Social Acceptance - Mid:lbnl_region'] == location]

    arr = newDF[scenario + ":" + attribute].to_numpy().astype(float)

    total = int(np.sum(arr))

    minVal = 0
    maxValue = total

    
    return dcc.Slider(
        id='slider', 
        min = 0,
        max = total,
        step= 1,
        value=total,
        tooltip={'placement':'bottom'}
    )



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
        fig.update_traces(marker_colorbar_title_text="", selector=dict(type='scattermapbox'))
        fig.update_layout(mapbox_style='open-street-map', legend_orientation="h")
        return dcc.Graph(
            figure=fig
        )



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
    elif(num_scenarios ==2):
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


def getViolinPlot(location, scenario, attribute):
    if(location=='All'):
        newDF = df.copy()
    else:
        newDF = df.loc[df['Existing Social Acceptance - Mid:lbnl_region'] == location]
    val = scenario+":"+attribute
    fig = px.violin(newDF, x=val, box=True, points="all")
    return fig

def getHistogramPlot(location,scenario,attribute):
    newDF = df.loc[df['Existing Social Acceptance - Mid:lbnl_region'] == location]
    val = scenario+":"+attribute
    fig = px.histogram(newDF, x=val)
    return fig
if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_ui=False,
                   dev_tools_props_check=False)
