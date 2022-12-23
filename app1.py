#!/usr/bin/python3

import numpy as np
import pandas as pd
import os 
import sys
import calendar
import datetime as dt
from datetime import date, timedelta, datetime
from shutil import get_terminal_size
pd.set_option('display.width', get_terminal_size()[0]) 
pd.set_option('display.max_columns', None)

import dash
from dash import Dash, html, dcc, Input, Output, ctx, State
import plotly.graph_objects as go
import plotly.express as px
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from plotly.subplots import make_subplots 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions=True 

df = pd.read_csv('merged_data2.csv');
df['Target Sales'] = 0.8*df['Sales Amount'] 

date = 'OrderDate'
df['date']=pd.to_datetime(df[date], infer_datetime_format=True).dt.date 
df = df.sort_values('date') 
dates = df['date'].unique()

customers = df['Name'].sort_values().unique()
regions = df['SalesTerritoryRegion'].sort_values().unique()
countries = df['SalesTerritoryCountry'].sort_values().unique()
EUR_to_USD = 0.9914

df.rename(columns={'SalesTerritoryCountry':'COUNTRY'}, inplace = True); #print(dfm)
dfcodes = pd.read_csv('2014_world_gdp_with_codes.csv');
df = df.merge(dfcodes,on = 'COUNTRY');

def main_table(data,cur,choice,tosum):
    dfg = data.groupby([choice]).count()['OrderQuantity'].to_frame(name = 'Items Sold').reset_index()
    df1 = data.sort_values(choice); #print(df1)
    names = dfg[choice].unique(); #print(names)
    df_price = pd.DataFrame()
    for (i,name) in enumerate(names):
        df_name = df1[df1[choice] == name]
        df_name.reset_index(drop=True, inplace=True)
        df_price = df_price.append(df_name.iloc[0])
    df_MT = dfg.merge(df_price,on = choice)
    df_MT['UnitPrice']= df_MT['UnitPrice']*df[cur]/EUR_to_USD
    df_MT['Sales Amount'] = df_MT['Items Sold']*df_MT['UnitPrice']
    df_MT = df_MT[[choice,'UnitPrice','Items Sold','Sales Amount']]
    df_MT['UnitPrice'] = round(df_MT['UnitPrice'],2);
    df_MT['Sales Amount'] = round(df_MT['Sales Amount'],1); #print(df_MT)
    df_MT['total'] = df_MT[tosum].sum()
    df_MT['Percent'] = round(100*df_MT[tosum]/df_MT['total'],2)
    return df_MT

def map_paras(data,para):
     dfm = data.groupby(['COUNTRY']).sum(); 
     dfM = dfm.merge(dfcodes,on = 'COUNTRY'); 
     return dfM

# For TABLE 2 ON 2ND PAGE

def table2_fix(data,cur):
    dfc = data.copy()
    dfc['UnitPrice']= data['UnitPrice']*data[cur]/EUR_to_USD
    dfo = dfc.groupby('Product Name').sum()['OrderQuantity'].to_frame(name = 'Items Sold').reset_index(); 
    dfu = dfc.groupby('Product Name').sum()['UnitPrice'].to_frame(name = 'Total Price').reset_index(); 
    names = dfc['Product Name'].unique(); #print(names)
    df_price = pd.DataFrame()
    for (i,name) in enumerate(names):
        df_name = data[data['Product Name'] == name]
        df_name.reset_index(drop=True, inplace=True)
        df_price = df_price.append(df_name.iloc[0])
    df_M = dfo.merge(df_price,on = 'Product Name'); #print(df_M)
    df_MT = df_M.merge(dfu,on = 'Product Name');
    df_MT['Total Price'] = round(df_MT['Total Price'],2);
    df_MT = df_MT[['Product Name','Description','Items Sold','Total Price']]; #print(df_MT)
    return df_MT

def sold(data):
    names = df['Product Name'].unique(); 
    df_sum = pd.DataFrame();
    for (i,name) in enumerate(names):
        df_name = data[data['Product Name'] == name]; #print(df_name)
        df_name = df_name[['Product Name','Color','Sales Amount','OrderQuantity']]
        df_name.reset_index(drop=True, inplace=True); #print(df_name)  
        df_name['Items Sold'] = df_name['OrderQuantity'].sum(); #print(df_name)
        df_sum =  df_sum.append(df_name.iloc[0]) # WANT SUMMARY TO RANK BY NO SOLD
        top = df_sum.sort_values('Items Sold',ascending=False);
    #print(top)
    return top
        
def time_series(data,n): # WILL USE ORIGINAL DATA, NO FILTERING BY CURRENCY OR DATE
    items = sold(df);# print(items)
    names = items['Product Name'].head(n); #print(names)
    df_top= pd.DataFrame(); 
    for (i,name) in enumerate(names):
        df_name = data[data['Product Name'] == name]; 
        df_name = df_name[['date','Product Name','Color','Sales Amount','OrderQuantity']]
        df_name.reset_index(drop=True, inplace=True);
        df_top =  df_top.append(df_name);
    #print(df_top)
    return df_top

#########################

SIDEBAR_STYLE = {"position": "fixed","top": 0,"left": 0,"bottom": 0,"width": "20rem",
                 "padding": "2rem 1rem","background-color": "#046F78",  "color": "white"}

CONTENT_STYLE = {"margin-left": "20rem", "margin-right": "0rem", "padding": "2rem 1rem"}

sidebar = html.Div([
    html.P(),     
        dbc.Nav(
            [
                dbc.NavLink("Overview", href="/", active="exact",
                    style={"background-color": "white",
                           "text-color": "black",'fontSize': 20}),
                html.P(), 
                dbc.NavLink("Top Selling Products", href="/top", active="exact",
                            style={"background-color": "white",
                                   "text-color": "black",'fontSize': 20})
            ],
            vertical=True,
            pills=False, # True hides inactive
        ),

 html.Br(),
    html.P("Date range"),
    
    html.Div(   # SO DROPDOWNS IN SAME ROW
        className="row",children=[
            html.Div(className='three columns', children=[
                dcc.Dropdown(
                    #['First','Last'],
                    dates,df[date].iat[0],
                    id='drop1',
                    style={'color': 'black', 'font-weight':'bold'}, 
                )], style=dict(width='50%')),
            
            html.Div(className='three columns', children=[
                dcc.Dropdown(
                   #nos, '1',
                   dates, df[date].iat[-1],
                    id='drop2',
                    style={'color': 'black','font-weight':'bold'},
                )], style=dict(width='50%')),
        ], style=dict(display='flex')
    ), # CLOSING DROPDOWNS

    dcc.Checklist(['No filter applied'], 
        id='check1',
        inline=True
        ),

    html.Br(),

   
    html.Button('AUD', id='btn-nclicks-1', n_clicks=0,style={'width':'53px','margin-right': '4px'}),
    html.Button('CAD', id='btn-nclicks-2', n_clicks=0,style={'width':'53px','margin-right': '4px'}),
    html.Button('EUR', id='btn-nclicks-3', n_clicks=0,style={'width':'53px','margin-right': '4px'}),
    html.Button('GBP', id='btn-nclicks-4', n_clicks=0,style={'width':'53px','margin-right': '4px'}),
    html.Button('USD', id='btn-nclicks-5', n_clicks=0,style={'width':'53px','margin-right': '4px'}),
    html.Div(id='container-button'),
    
    html.Br(),
                   
html.Hr(style = {"opacity": "unset"}), 

    html.Br(),
    html.Div(children=[
        html.Label(["Customer Name"], style={'fontSize': 18, "textAlign": "center" }),
        dcc.Dropdown(
            customers,
            "All",
            id='cus-drop',
            style={'color': 'black'}, #
    )
    ]),

    html.Br(),
    html.Div(children=[
        html.Label(["Region"], style={'fontSize': 18, "text-align": "center" }),
        dcc.Dropdown(
            regions, "All",
            id='reg-drop',
            style={'color': 'black'},
    ),
        ]),

    html.Br(),
    html.Div(children=[
        html.Label(["Country"], style={'fontSize': 18, "textAlign": "center" }),
        # NONE OF THIS CENTERING WORKING
        dcc.Dropdown(
            countries, "All",
            id='cou-drop',
            style={'color': 'black'},
    ),
        ]),
    
    
    ], style=dict(SIDEBAR_STYLE,overflow= "scroll"),
)

content = html.Div(id="page-content", style=CONTENT_STYLE)


app.layout = html.Div([dcc.Location(id="url"),sidebar,content])

@app.callback(Output("page-content", "children"),[Input("url", "pathname")])

def render_page_content(pathname):
    if pathname == "/":
         return html.Div([   
            html.P("Global Sales Analysis",
                   style={'font-family':'Arial Black','color': 'black', 'fontSize': 24,'textAlign':'center'}),
           
            dbc.Row(children = [   # OPEN TOP BUTTONS
                html.Div(children=[
                    dbc.Button('Total Sales',id='btn-map-1', className="mb-3",color="primary",n_clicks=0,
                               style={'margin-left': '250px','margin-right': '10px'}),  
                    dbc.Button('Units Sold',id='btn-map-2', className="mb-3",color="primary",n_clicks=0,
                               style={'margin-right': '10px'}),  
                    dbc.Button('Unit Price',id='btn-map-3', className="mb-3",color="primary",n_clicks=0,
                               style={'margin-right': '10px'}),
                    dbc.Button('Product Cost',id='btn-map-4', className="mb-3",color="primary",n_clicks=0,
                               style={'margin-right': '10px'})
                ]
                         ),
            ],
                    ), # CLOSE TOP BUTTONS

             html.Div([ # OPEN POP-UP
                 dcc.Graph(id='map1'),
        
                 dbc.Modal([
                     dbc.ModalBody([
                         dcc.Graph(id='gauge1'),
                         dbc.ModalBody(dbc.ModalTitle(id='text1',style={'fontSize': 30})),
                         dcc.Graph(id='yoy1'),
                     ]
                                   ),
                 ], id = 'modal1',
                           is_open=False,
                           ),
             ],
                      ),

             html.Div([   
                 dbc.Row([  # OPENING TABLE AND DONUT
                     dbc.Col([
                         dcc.Graph(id='table1'),
                     ]),
                     dbc.Col([
                         html.Div(id='textarea-output',style={'fontSize': 16,'font-weight':'bold'}),
                         dcc.Graph(id='donut1')
                     ]),
                 ]), # CLOSING TABLE AND DONUT
                 ])
         ]) # CLOSING HOME PAGE


    elif pathname == "/top":
        return html.Div([   
            html.P("Top Selling Products",
                   style={'font-family':'Arial Black','color': 'black', 'fontSize': 24,'textAlign':'center'}),
            html.Div(id='text2',style={'fontSize': 18,'font-weight':'bold'}),html.Br(),

            dcc.Graph(id='table2'),
            
            html.Div(id='text3',style={'fontSize': 18,'font-weight':'bold', 'margin-left': '100px'}),
            html.Br(),
            html.P([
                dbc.Row(children = [
                    "Select top n sellers",
                    dcc.Dropdown(
                        [1,5,10,20,50,100, "All"],
                        129, #ALL FOR DEFAULT
                        id='time-drop',
                        style={'width': '15%', 'margin-left': '10px', 'margin-right': '0px', 'font-weight':'bold'},
                    ),
                    "Show total ",
                        dcc.RadioItems(
                            ['Yes','No'],
                            'No', # default
                            id='total-show',
                            style={'width': '15%', 'margin-left': '0px', 'font-weight':'bold'},
                        ),
                    dbc.Col([
                        "Number of sales to smooth total by ",
                    html.Div([
                        dcc.Slider(
                            0,4,0.2,
                            id='slider',
                            marks={0: '1', 1: '10',2: '100', 3: '1000'}, #, 4: ''},
                            value=0
                        ),
                        ],style={'width': '45%', 'margin-left': '0px', 'font-weight':'bold'},
                             ),
                    ]),
                ]),
                dcc.Graph(id='time-series'),
            ],style={'margin-left': '60px'} 
                   )
    ])
                 

@app.callback(
    Output('map1', 'figure'),
    Output('table1', 'figure'),
    Output('donut1', 'figure'),
    Output('textarea-output', 'children'),
    
    Input('check1', 'value'), Input('drop1', 'value'), Input('drop2', 'value'),
    Input('btn-nclicks-1', 'n_clicks'),Input('btn-nclicks-2', 'n_clicks'),
    Input('btn-nclicks-3', 'n_clicks'),Input('btn-nclicks-4', 'n_clicks'),
    Input('btn-nclicks-5', 'n_clicks'),Input('btn-map-1', 'n_clicks'),
    Input('btn-map-2', 'n_clicks'), Input('btn-map-3', 'n_clicks'),
    Input('btn-map-4', 'n_clicks'),
    )

def some_plots(check,date1,date2,btn1,btn2,btn3,btn4,btn5,btn_map1,btn_map2,btn_map3,btn_map4):
       
    cur = 'USD';
        
    if "btn-nclicks-1" == ctx.triggered_id:
        cur = 'AUD'
              
    elif "btn-nclicks-2" == ctx.triggered_id:
        cur = 'CAD'
        
    elif "btn-nclicks-3" == ctx.triggered_id:
        cur = 'EUR'

    elif "btn-nclicks-4" == ctx.triggered_id:
        cur = 'GBP'
   
    elif "btn-nclicks-5" == ctx.triggered_id:
        cur = 'USD'

    df['Total Sales Amount'] = df['OrderQuantity']*df['UnitPrice']*df[cur]/EUR_to_USD   

    Date1 = datetime.strptime(date1,'%Y-%m-%d')
    Date2 = datetime.strptime(date2,'%Y-%m-%d')
    
    pd.options.mode.chained_assignment = None 
    df[date] = pd.to_datetime(df[date]);
 
    if check == None or check == []:  
        if Date1 < Date2:
            dff = df[df[date] >=  Date1]
            dff = dff[dff[date] <=  Date2]
           
        else:  # MAKING REVERSABLE
            dff = df[df[date] >=  Date2]
            dff = dff[dff[date] <=  Date1]
                   
    else:
        dff = df.copy()
        Date1 = df[date].iat[0]; 
        Date2= df[date].iat[-1];

    d1 = Date1.date(); d2 = Date2.date();
    
    choice = 'Product Name'
    
    df_T = main_table(dff,cur,choice,'Items Sold')
    cols = df_T.columns.values;

    table = go.Figure(data=[go.Table(
        columnwidth = [2.5,1,1],
        header=dict(values=([choice,'Items Sold','Sales Amount']),
                    fill_color='#305D91',
                    font=dict(color='white', size=14),
                    align='left'
                    ),
        cells=dict(values= [df_T['Product Name'],df_T['Items Sold'],df_T['Sales Amount']],
                   fill_color='#E0EEEF', #https://htmlcolorcodes.com
                   font=dict(color='black', size=14),
                   height=30, align='left'),
    )
                            ])

    table.update_layout(width=500, height=400,margin={"pad": 0, "t": 0,"r": 0,"l": 0,"b": 0})
  

    df_D = main_table(dff,cur,'Color','Items Sold'); #print(df_D)
    colours = df_D.iloc[:,0]; 

    donut = go.Figure(data=[go.Pie(labels=df_D['Color'],
                                   values=df_D['Items Sold'], hole=.4)])
    donut.update_traces(hoverinfo='label+percent', textinfo='value',
                        textfont_size=20, marker=dict(colors=colours,
                                                      line=dict(color='#000000', width=2)))
    donut.update_layout(width=400, height=400,margin={"pad": 0, "t": 0,"r": 50,"l": 20,"b": 0})

    para = 'Total Sales Amount'; prefix = cur
    
    if "btn-map-1" == ctx.triggered_id:
        para = 'Total Sales Amount'
        prefix = cur
        
    elif "btn-map-2" == ctx.triggered_id:
        para = 'OrderQuantity'
        prefix = ""
        
    elif "btn-map-3" == ctx.triggered_id:
        para = 'UnitPrice'
        prefix = cur

    elif "btn-map-4" == ctx.triggered_id:
        para = 'TotalProductCost'   
        prefix = cur
        
    df_M = map_paras(dff,para)
    world = go.Figure(data=go.Choropleth(locations = df_M['CODE'],z = df_M[para],
                                         text = df_M['COUNTRY'],colorscale = 'Rainbow',
                                         autocolorscale=False,reversescale=True,
                                         marker_line_color='black',
                                         marker_line_width=1,
                                         colorbar_tickprefix = prefix,
                                         ))
    world.update_layout(width=950, height=550,margin={"pad": 0, "t": 0,"r": 0,"l": 0,"b": 0},
                        clickmode='event', # 'event+select',
                        geo=dict(showframe=False, showcoastlines=False),
                        annotations = [dict(x=0.5,y=0.15, xref='paper',yref='paper',
                                            font=dict(color='black', size=16),
                                            text='%s by country over %s to %s'
                                            %(para,Date1.date(), Date2.date()),showarrow = False)]
                        ),
    world.update_traces(colorbar=dict(len=0.75))
    return world,table,donut,'Units Sold by Colour'


@app.callback(  
    Output('modal1', 'is_open'),
    Output('text1', 'children'),
    Output('gauge1', 'figure'),
    Output('yoy1', 'figure'), 
    [Input('map1', 'clickData')],
    [State("modal1", "is_open")]
    )

def popup(hoverData,is_open): 
    total = 0
    target = 0
    sales = 0
    gauge = go.Figure()
    text = ""
    yoy = px.scatter()
    yoy2 = px.scatter()
    d_y = pd.DataFrame()
    if hoverData != None: #and  n1:
        location = hoverData['points'][0]['location'] 
        z = hoverData['points'][0]['z'] 
        data = df[df['CODE'] == location]; d = data[data['Year'] == 2014]; #print(d)
        sales =  d['Sales Amount'].sum();
        text = "Total Sales -  %s" %(d['COUNTRY'].iat[0])
        target = 0.8*sales 
        ##### YoY GROWTH ##########
        #print(data)
            
        df_y = data.groupby(['Year']).sum()['Sales Amount'].to_frame(name = 'Total Sales').reset_index()
        yoy = make_subplots(specs=[[{"secondary_y": True}]]) ## Secondary axies

        yoy.add_trace(go.Scatter(x=df_y['Year'], y=df_y['Total Sales']),secondary_y=False)

        df_y['Percent Sales'] = 100*df_y['Total Sales']/df_y['Total Sales'].max()
        yoy.add_trace(go.Scatter(x=df_y['Year'], y=df_y['Percent Sales']), secondary_y=True)
        
        yoy.update_layout(width=450, height=350, margin={"pad": 0, "t": 0,"r": 0,"l": 0,"b": 0},
                          showlegend=False)
        yoy.update_xaxes(showgrid=False, linecolor = "black", linewidth = 3,mirror = True,
                     title="Year")
        yoy.update_yaxes(showgrid=False, linecolor = "black", linewidth = 3,mirror = False,
                         title="Sales [USD]",secondary_y=False)
          
        yoy.update_yaxes(showgrid=False, linecolor = "black", linewidth = 3,mirror = False,
                         title="Percentage of maximum", secondary_y=True)
        
        ############################
        gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta", value = sales,
            domain = {'x': [0.05, 0.95], 'y': [0.0, 1]},
            title = {'text': "2014 actual [%1.0f] vs. target [%1.0f] sales [USD]" %(sales,target),
                     'font': {'size': 18}},
            delta = {'reference': target, 'increasing': {'color': "red"}},
            gauge = {
                'axis': {'range': [None, 2*target], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},'bgcolor': "white",'borderwidth': 2,'bordercolor': "gray",
                'steps': [{'range': [0, target], 'color': 'red'}],
            }
    ),
                          )
        gauge.update_layout(font = {'color': "darkblue", 'family': "Arial"},
                            margin={"pad": 0, "t": 0,"r": 0,"l": 20,"b": 0})

        return not is_open,text,gauge,yoy
   
    return is_open,text,gauge,yoy
  

@app.callback(
    Output('table2', 'figure'),
    Output('text2', 'children'),Output('text3', 'children'),
    Input('check1', 'value'), Input('drop1', 'value'), Input('drop2', 'value'),
    Input('btn-nclicks-1', 'n_clicks'),Input('btn-nclicks-2', 'n_clicks'),
    Input('btn-nclicks-3', 'n_clicks'),Input('btn-nclicks-4', 'n_clicks'),
    Input('btn-nclicks-5', 'n_clicks')
    )

def page2(check,date1,date2,btn1,btn2,btn3,btn4,btn5):
    cur = 'USD';
        
    if "btn-nclicks-1" == ctx.triggered_id:
        cur = 'AUD'
              
    elif "btn-nclicks-2" == ctx.triggered_id:
        cur = 'CAD'
        
    elif "btn-nclicks-3" == ctx.triggered_id:
        cur = 'EUR'

    elif "btn-nclicks-4" == ctx.triggered_id:
        cur = 'GBP'
   
    elif "btn-nclicks-5" == ctx.triggered_id:
        cur = 'USD'

    df['Total Sales Amount'] = df['OrderQuantity']*df['UnitPrice']*df[cur]/EUR_to_USD   

    Date1 = datetime.strptime(date1,'%Y-%m-%d')
    Date2 = datetime.strptime(date2,'%Y-%m-%d')
    
    pd.options.mode.chained_assignment = None 
    df[date] = pd.to_datetime(df[date]);
 
    if check == None or check == []:  
        if Date1 < Date2:
            dff = df[df[date] >=  Date1]
            dff = dff[dff[date] <=  Date2]
           
        else:  # MAKING REVERSABLE
            dff = df[df[date] >=  Date2]
            dff = dff[dff[date] <=  Date1]
                   
    else:
        dff = df.copy()
        Date1 = df[date].iat[0]; 
        Date2= df[date].iat[-1];

    d1 = Date1.date(); d2 = Date2.date();
    
    df_T = table2_fix(dff,cur);
    tot_items = df_T['Items Sold'].sum()
    tot_sales = df_T['Total Price'].sum()
    
    table2 = go.Figure(data=[go.Table(
        columnwidth = [2,2,1,1],
        header=dict(values=(['Product Name','Description','Items Sold','Total Price']),
                    fill_color='#305D91',
                    font=dict(color='white', size=14),
                    align='left'
                    ),
        cells=dict(values=[df_T['Product Name'],df_T['Description'],df_T['Items Sold'],df_T['Total Price']],
                   fill_color='#E0EEEF', font=dict(color='black', size=14),
                   height=30, align='left'),
    )
                            ])

    table2.update_layout(width=900, height=650,margin={"pad": 0, "t": 0,"r": 0,"l": 50,"b": 0})

    
    return table2,'Sales over %s to %s in %s' %(Date1.date(), Date2.date(), cur),"Total.....................................................................................%d............ %1.0f" %(tot_items,tot_sales)


@app.callback(
    Output('time-series', 'figure'),
    Input('time-drop', 'value'),
    Input('total-show', 'value'),
    Input('slider', 'value')
)

def timeplot(n, showsum,slide_value):
    #print(df)
    if n == "All":
        n = 129
    
    print(n)
    dft = time_series(df,n)
    data = dft.copy()
    freq = data.groupby(['date','Product Name'])['OrderQuantity'].sum().reset_index(); 
    # NO WANT AVERAGE/TOTAL OF ALL ITEMS
    data['OrderQuantity'] = data['OrderQuantity'].astype(float); print(data)
    bunch = data.groupby(['date'])['OrderQuantity'].sum().reset_index(); print(bunch)
    slide_value = int(10**slide_value) # FOR LOG SCALE
    avg = bunch['OrderQuantity'].rolling(window=slide_value).mean()
    
    ts = px.line(freq, x="date", y="OrderQuantity", color='Product Name')
    if showsum == "Yes":
        ts.add_trace(go.Scatter(x=bunch.date, y=avg, showlegend=False,line=dict(width=2,color="black")))
        
    ts.update_xaxes(showgrid=False, linecolor = "black", linewidth = 2,mirror = True,title=" ")
    ts.update_yaxes(showgrid=False, linecolor = "black", linewidth = 2,mirror = True, title="Number of Items Sold")
    ts.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'},width=900,
                      margin={'l': 40, 'b': 40, 't': 20, 'r': 20}, font=dict(size=14),
                      hovermode='closest')

    
    return ts

      
if __name__ == '__main__':
    #app.run_server()
    app.run()
