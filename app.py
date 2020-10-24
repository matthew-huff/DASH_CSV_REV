import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import dash_table

dataFile = "../outfile_extended.csv"

diffDF = 0

DIFF_FN = ""
dataCSVs = {

    'All': pd.read_csv('../outfile_extended.csv'),
    'northeast': pd.read_csv('../data_extended_Northeast.csv'),
    'southeast': pd.read_csv('../data_extended_Southeast.csv'),
    'west': pd.read_csv('../data_extended_West.csv'),
    'great_lakes': pd.read_csv('../data_extended_Great_Lakes.csv'),
    'interior': pd.read_csv('../data_extended_Interior.csv')
}

df = dataCSVs['All']

locations = {'All': 'all',
             'Northeast': 'northeast',
             'Southeast': 'southeast',
             'West': 'west',
             'Great Lakes': 'great_lakes',
             'Interior': 'interior'}
locs = ['All', 'Northeast', 'Southeast', 'West', 'Great Lakes', 'Interior']

scenarios = ['Open Access - Mid', 'Open Access - Low', 'Open Access - Current', 'Baseline - Mid',
             'Baseline - Current', 'Baseline - Low', 'Legacy - Current', 'Existing Social Acceptance - Mid',
             'Mid Social Acceptance - Mid', 'Mid Social Acceptance - Low', 'Low Social Acceptance - Mid',
             'Low Social Acceptance - Low', 'Radar Limits - Mid',
             'Smart Bat Curtailment - Mid', 'Blanket Bat Curtailment - Mid',
             'Fed Land Exclusion - Mid', 'Limited Access - Mid']
dataFrame = dataCSVs['northeast']


colList = list(df.columns)
x = []
for col in colList:
    if(col.startswith('Existing Social Acceptance')):
        x.append(col)
attributes = [i.split(':')[1] for i in x]

cols = list(dataFrame.columns)
cols.remove('latitude')
cols.remove('longitude')
newCols = []
for col in cols:
    if(col.startswith('Existing Social Acceptance')):
        newCols.append(col)
cols = newCols


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets,
                suppress_callback_exceptions=True)

df = pd.read_csv(dataFile)

app.layout = html.Div(children=[

    html.H1(children="Wind Turbine Dashboard", style={
            'textAlign': 'center', 'padding': 10}),

    html.Div([

        html.Div([
            html.H4(children="Number of scenarios"),
            dcc.RadioItems(
                id='num_scenarios',
                options=[
                    {'label': '1', 'value': 1},
                    {'label': '2', 'value': 2}
                ],
                value=1
            ),
        ], className="four columns", style={'padding': 10}),

        html.Div([
            html.H4(children="LCOE Parameter"),
            dcc.RadioItems(
                id='lcoe_selection',
                options=[
                    {'label': 'Mean LCOE', 'value': 'mean_lcoe'},
                    {'label': 'Total LCOE', 'value': 'total_lcoe'}
                ],
                value='total_lcoe'
            ),
        ], className="four columns", style={'padding': 10}),
    ], className="row", style={'padding': 10}),


    html.Div(id='output'),



])


@app.callback(
    dash.dependencies.Output('output_curve_g1', 'children'),
    [dash.dependencies.Input('location_dropdown_g1', 'value'),
     dash.dependencies.Input('scenarios_g1', 'value'),
     dash.dependencies.Input('lcoe_selection', 'value')])
def update_g1(loc, scenario, lcoe):
    newDF = df.loc[df['Existing Social Acceptance - Mid:lbnl_region'] == loc]
    colList = list(newDF.columns)
    if(scenario == 'All'):
        return
    else:
        for col in colList:
            if(col != 'latitude' or col != 'longitude' or (not col.startswith(scenario))):
                colList.remove(col)
        newDF.drop(colList, axis=1)

    y_sorted = np.sort(newDF[scenario + ':' + lcoe].to_numpy().astype(float))
    fig = px.scatter(x=np.arange(len(y_sorted)),
                     y=np.flip(y_sorted), height=500)
    sum_y = np.nansum(y_sorted)
    return html.Div([
        dcc.Graph(
            id='g1',
            figure=fig
        ),
        html.Div([
            html.H5("Value under LCOE curve: " + format(int(sum_y), ','))
        ], style={'textAlign': 'center'})
    ])


@app.callback(
    dash.dependencies.Output('output_curve_g2', 'children'),
    [dash.dependencies.Input('location_dropdown_g2', 'value'),
     dash.dependencies.Input('scenarios_g2', 'value'),
     dash.dependencies.Input('lcoe_selection', 'value')])
def update_g2(loc, scenario, lcoe):
    newDF = df.loc[df['Existing Social Acceptance - Mid:lbnl_region'] == loc]
    colList = list(newDF.columns)
    if(scenario == 'All'):
        return
    else:
        for col in colList:
            if(col != 'latitude' or col != 'longitude' or (not col.startswith(scenario))):
                colList.remove(col)
        newDF.drop(colList, axis=1)

    y_sorted = np.sort(newDF[scenario + ':' + lcoe].to_numpy().astype(float))
    fig = px.scatter(x=np.arange(len(y_sorted)),
                     y=np.flip(y_sorted), height=500)
    sum_y = np.nansum(y_sorted)
    return html.Div([
        dcc.Graph(
            id='g2',
            figure=fig
        ),
        html.Div([
            html.H5("Value under LCOE curve: " + format(int(sum_y), ','))
        ], style={'textAlign': 'center'})

    ])


@app.callback(
    dash.dependencies.Output('output', 'children'),
    [dash.dependencies.Input('num_scenarios', 'value')])
def update_num_scenarios(num):

    if(num == 1):
        return html.Div([

            html.Div([


                html.Div([
                    html.H5("LBNL Region"),
                    dcc.Dropdown(
                        id='location_dropdown_g1',
                        options=[
                            {'label': l, 'value': l} for l in locs
                        ],
                        value='All'
                    ),
                ], className="four columns", style={'padding': 0}),


                html.Div([
                    html.H5("Scenarios"),
                    dcc.Dropdown(
                        id='scenarios_g1',
                        options=[
                            {'label': sc, 'value': sc} for sc in scenarios
                        ],
                        value='All'
                    ),
                ], className="four columns", style={'padding': 0})


            ], style={'padding': 10}),



            html.Div([
                html.Div(id='output_curve_g1', style={'padding': 150}),
                html.H5("Map Data Selection"),
                dcc.Dropdown(

                    id='map_data_selection_g1_selector',
                    options=[
                        {'label': col.split(':')[1], 'value':col.split(':')[1]} for col in cols
                    ],
                    value='mean_cf'
                ),


                html.Div(id='output_mapbox_g1'),

                html.H5("Correlation Table selections"),
                html.Div([
                    html.Div([
                        dcc.Dropdown(id='correlation_type_g1',
                                     options=[
                                         {'label': 'Pearson', 'value': 'pearson'},
                                         {'label': 'Kendall', 'value': 'kendall'},
                                         {'label': 'Spearman', 'value': 'spearman'}
                                     ],
                                     value='pearson')]),



                    html.Div(id="correlations_output_g1")
                ], className="row")
            ])
        ])

    elif(num == 2):
        return html.Div([html.Div([
            html.Div([
                html.H5("Location 1"),
                dcc.Dropdown(
                    id='location_dropdown_g1',
                    options=[
                        {'label': l, 'value': l} for l in locs
                    ],
                    value='All'
                ),
                html.H5("Scenario 1"),
                dcc.Dropdown(
                    id='scenarios_g1',
                    options=[
                        {'label': sc, 'value': sc} for sc in scenarios
                    ],
                    value='All'
                ),

            ], className="six columns", style={'padding': 10}),

            html.Div([
                html.H5("Location 2"),
                dcc.Dropdown(
                    id='location_dropdown_g2',
                    options=[
                        {'label': l, 'value': l} for l in locs
                    ],
                    value='All'
                ),
                html.H5("Scenario 2"),
                dcc.Dropdown(
                    id='scenarios_g2',
                    options=[
                        {'label': sc, 'value': sc} for sc in scenarios
                    ],
                    value='All'
                ),

            ], className="six columns", style={'padding': 10}),
        ], className="row", style={'padding': 10}),

            html.Div([
                html.Div([
                    html.Div(id='output_curve_g1'),
                    html.H5("Map Data Selection"),
                    dcc.Dropdown(

                        id='map_data_selection_g1_selector',
                        options=[
                            {'label': col.split(':')[1], 'value':col.split(':')[1]} for col in cols
                        ],
                        value='mean_cf'
                    ),

                    html.Div(id='output_mapbox_g1'),
                    html.H5("correlation Table selections"),
                    html.Div([
                        html.Div([
                            dcc.Dropdown(id='correlation_type_g1',
                                         options=[
                                             {'label': 'Pearson',
                                                 'value': 'pearson'},
                                             {'label': 'Kendall',
                                                 'value': 'kendall'},
                                             {'label': 'Spearman',
                                                 'value': 'spearman'}
                                         ],
                                         value='pearson')], className="six columns"),



                    ], className="row"),

                    html.Div(id="correlations_output_g1")
                ], className="six columns"),


                html.Div([
                    html.Div(id='output_curve_g2'),
                    html.H5("Map Data Selection"),
                    dcc.Dropdown(

                        id='map_data_selection_g2_selector',
                        options=[
                            {'label': col.split(':')[1], 'value':col.split(':')[1]} for col in cols
                        ],
                        value='mean_cf'
                    ),

                    html.Div(id='output_mapbox_g2'),

                    html.H5("correlation Table selections"),
                    html.Div([
                        html.Div([
                            dcc.Dropdown(id='correlation_type_g2',
                                         options=[
                                             {'label': 'Pearson',
                                                 'value': 'pearson'},
                                             {'label': 'Kendall',
                                                 'value': 'kendall'},
                                             {'label': 'Spearman',
                                                 'value': 'spearman'}
                                         ],
                                         value='pearson')], className="six columns"),



                    ], className="row"),

                    html.Div([


                        html.Div(id="correlations_output_g2",
                                 className='ten columns'),
                        html.Div([

                            dcc.Checklist(
                                id='corr_radio_list',
                                options=[{'label': i, 'value': i}
                                         for i in attributes],
                            )],
                            className='two columns'
                        ),

                    ])
                ], className="six columns")
            ], className="row"),

            html.Div([
                html.Div([
                    html.Div([
                        dcc.Dropdown(id='diff_selector',
                                     options=[
                                         {'label': col.split(':')[1], 'value':col.split(':')[1]} for col in cols
                                     ],
                                     value='Select value...'),
                    ], className="six columns"),

                    html.Div(
                        html.Button("Generate Diff", id='diff_button'),
                        className="six columns"
                    )

                ], style={'padding-top': 10}),




            ]),
            html.Div([
                html.Div([
                    html.H6("Diff file output = :")
                ], className="six columns"),
                html.Div([
                    html.H6(id="diff_filename")
                ], className="six columns")

            ]),

            html.Div(id='diff_mapbox_output', style={'padding-top': 80}),

            html.Div(id='empty')






        ])


@app.callback(
    dash.dependencies.Output('output_mapbox_g1', 'children'),
    [dash.dependencies.Input('location_dropdown_g1', 'value'),
     dash.dependencies.Input('scenarios_g1', 'value'),
     dash.dependencies.Input('num_scenarios', 'value'),
     dash.dependencies.Input('map_data_selection_g1_selector', 'value')])
def update_map_g1(loc, scenario, num_scenarios, data):
    size = 4
    zoom = 5
    newDF = df.loc[df['Existing Social Acceptance - Mid:lbnl_region'] == loc]
    if(num_scenarios == 1):
        height = 850
    elif(num_scenarios == 2):
        height = 600
    if(scenario == 'All'):
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


@app.callback(
    dash.dependencies.Output('output_mapbox_g2', 'children'),
    [dash.dependencies.Input('location_dropdown_g2', 'value'),
     dash.dependencies.Input('scenarios_g2', 'value'),
     dash.dependencies.Input('map_data_selection_g2_selector', 'value')])
def update_map_g2(loc, scenario, data):
    size = 4
    zoom = 5
    newDF = df.loc[df['Existing Social Acceptance - Mid:lbnl_region'] == loc]
    if(scenario == 'All'):
        pass
    else:
        if(loc == 'All'):
            size = 2
            zoom = 3

        size_arr = np.full(len(newDF[scenario+":"+data]), size)
        fig = px.scatter_mapbox(newDF, lat='latitude', lon='longitude', size=size_arr, hover_data=[
                                scenario+":"+data], size_max=size, color=scenario+":"+data, zoom=zoom, height=600)
        fig.update_layout(mapbox_style='open-street-map')
        return dcc.Graph(
            figure=fig
        )


@app.callback(
    dash.dependencies.Output('correlations_output_g1', 'children'),
    [dash.dependencies.Input('correlation_type_g1', 'value'),
     dash.dependencies.Input('location_dropdown_g1', 'value'),
     dash.dependencies.Input('scenarios_g1', 'value')]
)
def update_correlations_g1(typeCorr, loc, scenario):

    if(scenario == 'All'):
        return
    newDF = df.loc[df['Existing Social Acceptance - Mid:lbnl_region'] == loc]
    temp = newDF.copy()

    l = [x for x in list(temp.columns) if not x.startswith(scenario)]
    temp = temp.drop(l, axis=1)
    temp = temp.drop(scenario + ":count", axis=1)

    columns = list(temp.columns)
    columns = [i.split(':')[1] for i in columns]
    temp.columns = columns
    # temp.rename(str.split(':')[1], axis='columns')

    corr = temp.corr(typeCorr)
    corr = corr.round(4)
    cols = list(corr.columns)
    #
    corr.insert(0, " ", cols, False)
    # corr[" "] = cols

    return (dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in corr.columns],
        data=corr.to_dict('records'),
        fixed_columns={'headers': True, 'data': 1},
        style_table={'overflowX': 'auto', 'minWidth': '100%'}))


@app.callback(
    dash.dependencies.Output('correlations_output_g2', 'children'),
    [dash.dependencies.Input('correlation_type_g2', 'value'),
     dash.dependencies.Input('location_dropdown_g2', 'value'),
     dash.dependencies.Input('scenarios_g2', 'value'),
     dash.dependencies.Input('corr_radio_list', 'value')]
)
def update_correlations_g2(typeCorr, loc, scenario, corr_radio_list):

    if(scenario == 'All'):
        return
    newDF = df.loc[df['Existing Social Acceptance - Mid:lbnl_region'] == loc]
    temp = newDF.copy()

    l = [x for x in list(temp.columns) if not x.startswith(scenario)]
    temp = temp.drop(l, axis=1)
    temp = temp.drop(scenario + ":count", axis=1)

    if(type(corr_radio_list) == type(None)):
        pass
    elif(type(corr_radio_list) == list and len(corr_radio_list) == 0):
        pass
    else:
        dropList = [scenario + ':' + i for i in corr_radio_list]
        temp = temp.drop(dropList, axis=1)

    columns = list(temp.columns)
    columns = [i.split(':')[1] for i in columns]
    temp.columns = columns
    # temp.rename(str.split(':')[1], axis='columns')

    corr = temp.corr(typeCorr)
    cols = list(corr.columns)
    #
    corr.insert(0, " ", cols, False)
    # corr[" "] = cols

    return html.Div([

        html.Div((dash_table.DataTable(
            id='table',
            columns=[{"name": i, "id": i} for i in corr.columns],
            data=corr.to_dict('records'),
            fixed_columns={'headers': True, 'data': 1},
            style_table={'overflowX': 'auto', 'minWidth': '100%'})), className="ten columns"),



    ], className='row')


@ app.callback(
    dash.dependencies.Output('corr_radio', 'children'),
    [dash.dependencies.Input('empty', 'children')]
)
def corr_radio_items(empty):
    colList = list(df.columns)
    x = []
    for col in colList:
        if(col.startswith('Existing Social Acceptance')):
            x.append(col)
    columns = [i.split(':')[1] for i in x]

    return


@ app.callback(
    dash.dependencies.Output('diff_mapbox_output', 'children'),
    [dash.dependencies.Input('diff_selector', 'value')],
    [dash.dependencies.State('diff_filename', 'children'),
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
    dash.dependencies.Output('diff_filename', 'children'),
    [
        dash.dependencies.Input('diff_button', 'n_clicks'),
    ],
    state=[
        dash.dependencies.State('location_dropdown_g1', 'value'),
        dash.dependencies.State('location_dropdown_g2', 'value'),
        dash.dependencies.State('scenarios_g1', 'value'),
        dash.dependencies.State('scenarios_g2', 'value'),
        dash.dependencies.State('diff_selector', 'value')
    ]
)
def generate_diff(n_clicks, loc1, loc2, scen1, scen2, diff_s):
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
            wr(var)
            continue

        try:
            diff_data = np.subtract(scen2_data.to_numpy().astype(
                float), scen1_data.to_numpy().astype(float))
            diff_dict[var] = diff_data
        except TypeError:
            pass

    diffDF = pd.DataFrame.from_dict(diff_dict, dtype='str')

    diffDF.to_csv(DIFF_FN)
    diffDF = pd.read_csv(DIFF_FN)
    return DIFF_FN


def wr(output):
    f = open('log.txt', 'a')
    f.write("----------\n")
    f.write(output)
    f.close()


"""
@app.callback(
    dash.dependencies.Output('map_data_selection_g1', 'children')
    [dash.dependencies.Input('none', 'children')]
)
def map_selector_g1():
    cols = list(dataFrame.columns)
    for col in cols:
        if(not col.startswith('Existing Social Acceptance')):
            cols.remove(col)

    return dcc.Dropdown(
                id='map_data_g1_selector',
                options=[
                    {'label':col.split(':')[1], 'value':col.split(':')[1]} for col in cols
                ],
                value='mean_cf'
            ),

@app.callback(
    dash.dependencies.Output('map_data_selection_g2', 'children')
)
def map_selector_g2():
    return returnMapChoiceSelector('map_data_selection_g2_selector')


def returnMapChoiceSelector(id_val):
    cols = list(dataFrame.columns)
    for col in cols:
        if(not col.startswith('Existing Social Acceptance')):
            cols.remove(col)

    return dcc.Dropdown(
                id=id_val,
                options=[
                    {'label':col.split(':')[1], 'value':col.split(':')[1]} for col in cols
                ],
                value='mean_cf'
            ),


"""


if __name__ == '__main__':
    app.run_server(debug=True)
