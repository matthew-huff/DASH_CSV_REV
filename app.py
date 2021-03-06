import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import dash_table
import time

dataFile = "../outfile_extended.csv"

diffDF = 0
devDF = 0
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

currentSelectors = 0
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

        html.Div([
            html.H4("View"),
            dcc.RadioItems(
                id='view_selector',
                options=[
                    {'label': 'Data', 'value': 'Data'},
                    {'label': 'Deviations', 'value': 'Deviations'}
                ],
                value='Data'
            )
        ])
    ], className="row", style={'padding': 10}),

    html.Div(id='selector'),
    html.Div(id='output'),



])


@app.callback(
    dash.dependencies.Output('output_curve_g1', 'children'),
    [dash.dependencies.Input('num_scenarios', 'value'),
     dash.dependencies.Input('location_dropdown_g1', 'value'),
     dash.dependencies.Input('scenarios_g1', 'value'),
     dash.dependencies.Input('lcoe_selection', 'value')])
def update_g1(num, loc, scenario, lcoe):
    newDF = df.loc[df['Existing Social Acceptance - Mid:lbnl_region'] == loc]
    colList = list(newDF.columns)
    if(num == 1):
        height = 850
    elif(num == 2):
        height = 500
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
        ], style={'textAlign': 'left'})

    ])


@app.callback(
    dash.dependencies.Output('one', 'children'),
    [dash.dependencies.Input('num_scenarios', 'value')]
)
def g1(num):

    if(num == 1):
        return html.Div([
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

                html.Div([
                    html.Div(id="correlations_output_g1",
                             className='ten columns'),
                    html.Div([
                        html.H5("Remove"),
                        dcc.Checklist(
                            id='corr_radio_list_g1',
                            options=[{'label': i, 'value': i}
                                     for i in attributes],
                        )],
                        className='two columns'
                    ),

                ])
            ], className="row")])
    elif(num == 2):
        return html.Div([html.Div([
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

                html.Div([
                    html.Div(id="correlations_output_g1",
                             className='ten columns'),
                    html.Div([
                        html.H5("Remove"),
                        dcc.Checklist(
                            id='corr_radio_list_g1',
                            options=[{'label': i, 'value': i}
                                     for i in attributes],
                        )],
                        className='two columns'
                    ),

                ])
            ], className="row")]),

            html.Div([


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
                html.Div(id='empty')], className="row")])


@ app.callback(
    dash.dependencies.Output('two', 'children'),
    [dash.dependencies.Input('num_scenarios', 'value')]
)
def g2(num):
    return html.Div([
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
                    html.H5("Remove"),
                    dcc.Checklist(
                        id='corr_radio_list_g2',
                        options=[{'label': i, 'value': i}
                                 for i in attributes],
                    )],
                    className='two columns'
                ),

            ])
        ], className="row")])


@ app.callback(
    dash.dependencies.Output('selector', 'children'),
    [dash.dependencies.Input('num_scenarios', 'value'),
     dash.dependencies.Input('view_selector', 'value')]
)
def update_selectors(num, view):
    if(view == 'Data'):
        if(num == 1):
            return html.Div([
                html.Div([
                    html.H5("Location"),
                    dcc.Dropdown(
                        id='location_dropdown_g1',
                        options=[
                            {'label': l, 'value': l} for l in locs
                        ],
                        value='All'
                    ),
                    html.H5("Scenario"),
                    dcc.Dropdown(
                        id='scenarios_g1',
                        options=[
                            {'label': sc, 'value': sc} for sc in scenarios
                        ],
                        value='All'
                    ),
                ], className='six columns'), ], className='row')

        elif(num == 2):
            return html.Div([
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
                ], className='six columns'),
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
                    )
                ], className='six columns')])
    elif(view == 'Deviations'):
        return html.Div([
            html.Div(id='dev_selector', className="two columns"),
            html.Div(id='deviations_view', className="ten columns")], className="row")


@ app.callback(
    dash.dependencies.Output('output', 'children'),
    [dash.dependencies.Input('num_scenarios', 'value'),
     dash.dependencies.Input('view_selector', 'value')]
)
def update_num_scenarios(num, view):
    if(view == 'Data'):
        if(num == 1):
            return html.Div([
                html.Div(id='one')
            ], style={'padding': 10})

        elif(num == 2):
            return html.Div([
                html.Div([
                    html.Div(
                        id='one',
                        className='six columns'
                    ),
                    html.Div(
                        id='two',
                        className='six columns'
                    )
                ])])


@ app.callback(
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


@ app.callback(
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


@ app.callback(
    dash.dependencies.Output('correlations_output_g1', 'children'),
    [dash.dependencies.Input('correlation_type_g1', 'value'),
     dash.dependencies.Input('location_dropdown_g1', 'value'),
     dash.dependencies.Input('scenarios_g1', 'value'),
     dash.dependencies.Input('corr_radio_list_g1', 'value')]
)
def update_correlations_g1(typeCorr, loc, scenario, corr_radio_list):

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


@ app.callback(
    dash.dependencies.Output('correlations_output_g2', 'children'),
    [dash.dependencies.Input('correlation_type_g2', 'value'),
     dash.dependencies.Input('location_dropdown_g2', 'value'),
     dash.dependencies.Input('scenarios_g2', 'value'),
     dash.dependencies.Input('corr_radio_list_g2', 'value')]
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
    corr = corr.round(4)
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
                    {'label': i, 'value': i} for i in scenarios
                ],
                value=[i for i in scenarios],

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
    avgDict['lbnl_regions'] = newDF['Existing Social Acceptance - Mid:lbnl_region']
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


@ app.callback(
    dash.dependencies.Output('dev_data_std', 'children'),
    [dash.dependencies.Input('dev_selector_checklist', 'value'),
     dash.dependencies.Input('dev_dropdown_std', 'value')]
)
def generate_deviations_std(selectors, dropdown):
    global currentSelectors
    global devDF
    if(selectors != currentSelectors):
        currentSelectors = selectors
        generate_deviations_DF(selectors, dropdown)
    time.sleep(4)
    size_arr = devDF['num_scenarios'].to_numpy()
    fig = px.scatter_mapbox(devDF, lat='latitude', lon='longitude', size=size_arr,
                            size_max=3, color=dropdown, zoom=3, height=850)
    fig.update_layout(mapbox_style='open-street-map')

    return html.Div([
        html.Div([
            dcc.Graph(
                 figure=fig
                 )
        ])

    ])


@ app.callback(
    dash.dependencies.Output('dev_data_q', 'children'),
    [dash.dependencies.Input('dev_selector_checklist', 'value'),
     dash.dependencies.Input('dev_dropdown_q', 'value')]
)
def generate_deviations_q(selectors, dropdown):
    global currentSelectors
    global devDF
    if(selectors != currentSelectors):
        currentSelectors = selectors
        generate_deviations_DF(selectors, dropdown)

    size_arr = devDF['num_scenarios'].to_numpy()
    fig = px.scatter_mapbox(devDF, lat='latitude', lon='longitude', size=size_arr,
                            size_max=3, color=dropdown, zoom=3, height=850)
    fig.update_layout(mapbox_style='open-street-map')
    return html.Div([
        html.Div([
            dcc.Graph(
                figure=fig
            )
        ])

    ])


def wr(output):
    f = open('log.txt', 'a')
    f.write("----------\n")
    f.write(output)
    f.close()


if __name__ == '__main__':
    app.run_server(debug=True)
