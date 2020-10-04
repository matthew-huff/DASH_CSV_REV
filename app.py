import dash
import dash_core_components as dcc 
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go 
import numpy as np
import pandas as pd 
import dash_table

dataFile = "../outfile_extended.csv"

dataCSVs = {
    
    'all' : pd.read_csv('../outfile_extended.csv'),
    'northeast' : pd.read_csv('../data_extended_Northeast.csv'),
    'southeast' : pd.read_csv('../data_extended_Southeast.csv'),
    'west' : pd.read_csv('../data_extended_West.csv'),
    'great_lakes': pd.read_csv('../data_extended_Great_Lakes.csv'),
    'interior' : pd.read_csv('../data_extended_Interior.csv')
}
dataFrame = dataCSVs['northeast']

cols = list(dataFrame.columns)
cols.remove('latitude')
cols.remove('longitude')
newCols = []
for col in cols:
    if(col.startswith('Existing Social Acceptance')):
        newCols.append(col)
cols = newCols


  
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

df = pd.read_csv(dataFile)

app.layout = html.Div(children=[
    html.H1(children="LCOE Curves"),
    
    html.H4(children="Number of scenarios"),
    html.Div([
        dcc.Dropdown(
            id='num_scenarios',
            options=[
                {'label': '1', 'value': 1},
                {'label': '2', 'value': 2}
            ],
            value=1
        ),
    ], style={'padding': 10}),
    
    html.Div([
        dcc.Dropdown(
            id='lcoe_selection',
            options=[
                {'label': 'Mean LCOE', 'value': 'mean_lcoe'},
                {'label': 'Total LCOE', 'value': 'total_lcoe'}
            ],
            value='total_lcoe'
        ),
    ], style={'padding': 10}),
    
    html.Div(id='output'),
        
    
 
])


@app.callback(
    dash.dependencies.Output('output_curve_g1', 'children'),
    [dash.dependencies.Input('location_dropdown_g1', 'value'),
     dash.dependencies.Input('scenarios_g1', 'value'),
     dash.dependencies.Input('lcoe_selection', 'value')])
def update_g1(loc, scenario, lcoe):
    df = dataCSVs[loc]
    colList = list(df.columns)
    if(scenario == 'all'):
        scenario = 'Existing Social Acceptance - Mid'
    else:
        for col in colList:
            if(col != 'latitude' or col != 'longitude' or (not col.startswith(scenario))):
                colList.remove(col)
        df.drop(colList, axis=1)
        


    y_sorted = np.sort(df[ scenario + ':' + lcoe].to_numpy().astype(float))
    fig = px.scatter(x=np.arange(len(y_sorted)),
            y=np.flip(y_sorted), height=600)
    sum_y = np.nansum(y_sorted)
    return html.Div([
        dcc.Graph(
            id='g1',
            figure=fig
        ),
        html.Div([
            html.H5("Value under LCOE curve: " + format(int(sum_y), ','))
        ],style={'textAlign': 'center'})
    ])
    

@app.callback(
    dash.dependencies.Output('output_curve_g2', 'children'),
    [dash.dependencies.Input('location_dropdown_g2', 'value'),
     dash.dependencies.Input('scenarios_g2', 'value'),
     dash.dependencies.Input('lcoe_selection', 'value')])
def update_g2(loc, scenario, lcoe):
    df = dataCSVs[loc]
    colList = list(df.columns)
    if(scenario == 'all'):
        scenario = 'Existing Social Acceptance - Mid'
    else:
        for col in colList:
            if(col != 'latitude' or col != 'longitude' or (not col.startswith(scenario))):
                colList.remove(col)
        df.drop(colList, axis=1)
        


    y_sorted = np.sort(df[ scenario +  ':' + lcoe].to_numpy().astype(float))
    fig = px.scatter(x=np.arange(len(y_sorted)),
            y=np.flip(y_sorted), height=600)
    sum_y = np.nansum(y_sorted)
    return html.Div([
        dcc.Graph(
            id='g2',
            figure=fig
        ),
        html.Div([
            html.H5("Value under LCOE curve: " + format(int(sum_y), ','))
        ],style={'textAlign': 'center'})
        
    ])


@app.callback(
    dash.dependencies.Output('output', 'children'),
    [dash.dependencies.Input('num_scenarios', 'value')])
def update_num_scenarios(num):

    if(num==1):
        return html.Div([
            
            html.Div([
             dcc.Dropdown(   
                id='location_dropdown_g1',
                options=[
                    {'label': 'All', 'value': 'all'},
                    {'label': 'Northeast', 'value': 'northeast'},
                    {'label': 'Southeast', 'value': 'southeast'},
                    {'label': 'Great Lakes', 'value': 'great_lakes'},
                    {'label': 'West', 'value': 'west'},
                    {'label': 'Interior', 'value': 'interior'}
                ],
                value='all'
            ),

            dcc.Dropdown(   
                    id='scenarios_g1',
                    options=[
                        {'label': 'All', 'value' : 'all'},
                        {'label': 'Open Access - Mid', 'value': 'Open Access - Mid'},
                        {'label': 'Open Access - Low', 'value': 'Open Access - Low'},
                        {'label': 'Open Access - Current', 'value': 'Open Access - Current'},
                        {'label': 'Baseline - Current', 'value': 'Baseline - Current'},
                        {'label': 'Baseline - Low', 'value': 'Baseline - Low'},
                        {'label': 'Legacy - Current', 'value': 'Legacy - Current'},
                        {'label': 'Existing Social Acceptance - Mid', 'value': 'Existing Social Acceptance - Mid'},
                        {'label': 'Mid Social Acceptance - Mid', 'value': 'Mid Social Acceptance - Mid'},
                        {'label': 'Mid Social Acceptance - Low', 'value': 'Mid Social Acceptance - Low'},
                        {'label': 'Low Social Acceptance - Mid', 'value': 'Low Social Acceptance - Mid'},
                        {'label': 'Low Social Acceptance - Low', 'value': 'Low Social Acceptance - Low'},
                        {'label': 'Radar Limits - Mid', 'value': 'Radar Limits - Mid'},
                        {'label': 'Smart Bat Curtailment - Mid', 'value': 'Smart Bat Curtailment - Mid'},
                        {'label': 'Blanket Bat Curtailment - Mid', 'value': 'Blanket Bat Curtailment - Mid'},
                        {'label': 'Fed Land Exclusion - Mid', 'value': 'Fed Land Exclusion - Mid'},
                        {'label': 'Limited Access - Mid', 'value': 'Limited Access - Mid'}
                    ],
                    value='all'
                ),

        ],style={'padding': 10}),

        
       
        html.Div([
            html.Div(id='output_curve_g1'),
            html.H5("Map Data Selection"),
            dcc.Dropdown(   
                
                id='map_data_selection_g1_selector',
                options=[
                    {'label':col.split(':')[1], 'value':col.split(':')[1]} for col in cols
                ],
                value='mean_cf'
            ),
            

            html.Div(id='output_mapbox_g1'),

            html.H5("Correlation Table selections"),
            html.Div([
                html.Div([
                    dcc.Dropdown(id='correlation_type_g1',
                    options=[
                        {'label' : 'Pearson', 'value':'pearson'},
                        {'label' : 'Kendall', 'value': 'kendall'},
                        {'label' : 'Spearman', 'value': 'spearman'}
                    ],
                    value='pearson')]),
                
                html.Div([
                    dcc.Input(
                        id="min_periods_g1",
                        type='number',
                        placeholder="min_periods >= 1",
                        value=1
                    )
                ]),

                html.Div(id="correlations_output_g1")
        ], className="row")
        ]) 
    ])
    
    elif(num==2):
        return html.Div([ html.Div([
        html.Div([
             dcc.Dropdown(   
                id='location_dropdown_g1',
                options=[
                    {'label': 'All', 'value': 'all'},
                    {'label': 'Northeast', 'value': 'northeast'},
                    {'label': 'Southeast', 'value': 'southeast'},
                    {'label': 'Great Lakes', 'value': 'great_lakes'},
                    {'label': 'West', 'value': 'west'},
                    {'label': 'Interior', 'value': 'interior'}
                ],
                value='all'
            ),

            dcc.Dropdown(   
                    id='scenarios_g1',
                    options=[
                        {'label': 'All', 'value' : 'all'},
                        {'label': 'Open Access - Mid', 'value': 'Open Access - Mid'},
                        {'label': 'Open Access - Low', 'value': 'Open Access - Low'},
                        {'label': 'Open Access - Current', 'value': 'Open Access - Current'},
                        {'label': 'Baseline - Current', 'value': 'Baseline - Current'},
                        {'label': 'Baseline - Low', 'value': 'Baseline - Low'},
                        {'label': 'Legacy - Current', 'value': 'Legacy - Current'},
                        {'label': 'Existing Social Acceptance - Mid', 'value': 'Existing Social Acceptance - Mid'},
                        {'label': 'Mid Social Acceptance - Mid', 'value': 'Mid Social Acceptance - Mid'},
                        {'label': 'Mid Social Acceptance - Low', 'value': 'Mid Social Acceptance - Low'},
                        {'label': 'Low Social Acceptance - Mid', 'value': 'Low Social Acceptance - Mid'},
                        {'label': 'Low Social Acceptance - Low', 'value': 'Low Social Acceptance - Low'},
                        {'label': 'Radar Limits - Mid', 'value': 'Radar Limits - Mid'},
                        {'label': 'Smart Bat Curtailment - Mid', 'value': 'Smart Bat Curtailment - Mid'},
                        {'label': 'Blanket Bat Curtailment - Mid', 'value': 'Blanket Bat Curtailment - Mid'},
                        {'label': 'Fed Land Exclusion - Mid', 'value': 'Fed Land Exclusion - Mid'},
                        {'label': 'Limited Access - Mid', 'value': 'Limited Access - Mid'}
                    ],
                    value='all'
                ),

        ], className="six columns", style={'padding': 10}),

        html.Div([
            dcc.Dropdown(   
                id='location_dropdown_g2',
                options=[
                    {'label': 'All', 'value': 'all'},
                    {'label': 'Northeast', 'value': 'northeast'},
                    {'label': 'Southeast', 'value': 'southeast'},
                    {'label': 'Great Lakes', 'value': 'great_lakes'},
                    {'label': 'West', 'value': 'west'},
                    {'label': 'Interior', 'value': 'interior'}
                ],
                value='all'
            ),

            dcc.Dropdown(   
                    id='scenarios_g2',
                    options=[
                        {'label': 'All', 'value' : 'all'},
                        {'label': 'Open Access - Mid', 'value': 'Open Access - Mid'},
                        {'label': 'Open Access - Low', 'value': 'Open Access - Low'},
                        {'label': 'Open Access - Current', 'value': 'Open Access - Current'},
                        {'label': 'Baseline - Current', 'value': 'Baseline - Current'},
                        {'label': 'Baseline - Low', 'value': 'Baseline - Low'},
                        {'label': 'Legacy - Current', 'value': 'Legacy - Current'},
                        {'label': 'Existing Social Acceptance - Mid', 'value': 'Existing Social Acceptance - Mid'},
                        {'label': 'Mid Social Acceptance - Mid', 'value': 'Mid Social Acceptance - Mid'},
                        {'label': 'Mid Social Acceptance - Low', 'value': 'Mid Social Acceptance - Low'},
                        {'label': 'Low Social Acceptance - Mid', 'value': 'Low Social Acceptance - Mid'},
                        {'label': 'Low Social Acceptance - Low', 'value': 'Low Social Acceptance - Low'},
                        {'label': 'Radar Limits - Mid', 'value': 'Radar Limits - Mid'},
                        {'label': 'Smart Bat Curtailment - Mid', 'value': 'Smart Bat Curtailment - Mid'},
                        {'label': 'Blanket Bat Curtailment - Mid', 'value': 'Blanket Bat Curtailment - Mid'},
                        {'label': 'Fed Land Exclusion - Mid', 'value': 'Fed Land Exclusion - Mid'},
                        {'label': 'Limited Access - Mid', 'value': 'Limited Access - Mid'}
                    ],
                    value='all'
                ),

        ], className="six columns", style={'padding' : 10}),
        ], className="row", style={'padding' : 10}),
       
    html.Div([
        html.Div([
            html.Div(id='output_curve_g1'),
            html.H5("Map Data Selection"),
            dcc.Dropdown(   
                
                id='map_data_selection_g1_selector',
                options=[
                    {'label':col.split(':')[1], 'value':col.split(':')[1]} for col in cols
                ],
                value='mean_cf'
            ),
            
            html.Div(id='output_mapbox_g1'),
            html.H5("correlation Table selections"),
            html.Div([
                html.Div([
                    dcc.Dropdown(id='correlation_type_g1',
                    options=[
                        {'label' : 'Pearson', 'value':'pearson'},
                        {'label' : 'Kendall', 'value': 'kendall'},
                        {'label' : 'Spearman', 'value': 'spearman'}
                    ],
                    value='pearson')], className="six columns"),
                
                html.Div([
                    dcc.Input(
                        id="min_periods_g1",
                        type='number',
                        placeholder="min_periods >= 1",
                        value=1
                    )
                ], className="six columns")
                
            ], className="row"),
            
            html.Div(id="correlations_output_g1")
        ], className="six columns"),
        

        html.Div([
            html.Div(id='output_curve_g2'),
            html.H5("Map Data Selection"),
            dcc.Dropdown(   
                
                id='map_data_selection_g2_selector',
                options=[
                    {'label':col.split(':')[1], 'value':col.split(':')[1]} for col in cols
                ],
                value='mean_cf'
            ),
            
            html.Div(id='output_mapbox_g2'),

            html.H5("correlation Table selections"),
            html.Div([
                html.Div([
                    dcc.Dropdown(id='correlation_type_g2',
                    options=[
                        {'label' : 'Pearson', 'value':'pearson'},
                        {'label' : 'Kendall', 'value': 'kendall'},
                        {'label' : 'Spearman', 'value': 'spearman'}
                    ],
                    value='pearson')], className="six columns"),
                
                html.Div([
                    dcc.Input(
                        id="min_periods_g2",
                        type='number',
                        placeholder="min_periods >= 1",
                        value=1
                    )
                ], className="six columns")
                
            ], className="row"),
            
            html.Div(id="correlations_output_g2")
        ], className="six columns")
    ], className="row")
    

         ])


@app.callback(
    dash.dependencies.Output('output_mapbox_g1', 'children'),
    [dash.dependencies.Input('location_dropdown_g1', 'value'),
     dash.dependencies.Input('scenarios_g1', 'value'),
     dash.dependencies.Input('num_scenarios', 'value'),
     dash.dependencies.Input('map_data_selection_g1_selector', 'value')])
def update_map_g1(loc, scenario, num_scenarios, data):
    
    df = dataCSVs[loc]
    if(num_scenarios==1):
        height=850
    elif(num_scenarios==2):
        height=600
    if(scenario == 'all'):
        pass
    else:
        size = 4
        size_arr = np.full( len(df[scenario+":"+data]), size)
        fig = px.scatter_mapbox(df, lat='latitude', lon='longitude', size=size_arr, hover_data=[scenario+":"+data], size_max=size, color=scenario+":"+data, zoom=5, height=height)
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
    df = dataCSVs[loc]

    if(scenario == 'all'):
        pass
    else:
        size = 4
        size_arr = np.full( len(df[scenario+':' + data]), size)
        fig = px.scatter_mapbox(df, lat='latitude', lon='longitude', size=size_arr, hover_data=[scenario+':'+data], size_max=size, color=scenario+':'+data, zoom=5, height=600)
        fig.update_layout(mapbox_style='open-street-map')
        return dcc.Graph(
            figure=fig
        )


@app.callback(
    dash.dependencies.Output('correlations_output_g1', 'children'),
    [dash.dependencies.Input('correlation_type_g1', 'value'),
     dash.dependencies.Input('min_periods_g1', 'value'),
     dash.dependencies.Input('location_dropdown_g1', 'value'),
     dash.dependencies.Input('scenarios_g1', 'value')]
)
def update_correlations_g1(typeCorr, min_period, loc, scenario):

    if(scenario == 'all'):
        return
    df = dataCSVs[loc]
    temp = df.copy()

    l = [x for x in list(temp.columns) if not x.startswith(scenario)]
    temp = temp.drop(l, axis=1)
    temp = temp.drop(scenario + ":count", axis=1)

    columns = list(temp.columns)
    columns = [i.split(':')[1] for i in columns]
    temp.columns = columns
    #temp.rename(str.split(':')[1], axis='columns')
    
    corr = temp.corr(typeCorr, min_period)
    cols = list(corr.columns)
    #
    corr.insert(0, " ", cols, False)
    #corr[" "] = cols
    
    return (dash_table.DataTable(
    id='table',
    columns=[{"name": i, "id": i} for i in corr.columns],
    data= corr.to_dict('records'),
    fixed_columns={'headers':True, 'data': 1},
    style_table={'overflowX': 'auto', 'minWidth': '100%'}))


@app.callback(
    dash.dependencies.Output('correlations_output_g2', 'children'),
    [dash.dependencies.Input('correlation_type_g2', 'value'),
     dash.dependencies.Input('min_periods_g2', 'value'),
     dash.dependencies.Input('location_dropdown_g2', 'value'),
     dash.dependencies.Input('scenarios_g2', 'value')]
)
def update_correlations_g2(typeCorr, min_period, loc, scenario):

    if(scenario == 'all'):
        return
    df = dataCSVs[loc]
    temp = df.copy()

    l = [x for x in list(temp.columns) if not x.startswith(scenario)]
    temp = temp.drop(l, axis=1)
    temp = temp.drop(scenario + ":count", axis=1)

    columns = list(temp.columns)
    columns = [i.split(':')[1] for i in columns]
    temp.columns = columns
    #temp.rename(str.split(':')[1], axis='columns')

    corr = temp.corr(typeCorr, min_period)
    cols = list(corr.columns)
    #
    corr.insert(0, " ", cols, False)
    #corr[" "] = cols
    
    return (dash_table.DataTable(
    id='table',
    columns=[{"name": i, "id": i} for i in corr.columns],
    data= corr.to_dict('records'),
    fixed_columns={'headers':True, 'data': 1},
    style_table={'overflowX': 'auto', 'minWidth': '100%'}))



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

