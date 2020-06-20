import os
import pandas as pd
import dash
import dash_core_components as dcc
from dash.dependencies import Input, Output
import dash_html_components as html
import plotly.graph_objects as go
import plotly.express as px
import dash_table as dt
import warnings
from fbprophet import Prophet


warnings.simplefilter(action='ignore')

css = ["https://stackpath.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css"]

app = dash.Dash(__name__, external_stylesheets=css)
fig = go.Figure()
# Data manipulation
S = os.path.dirname(os.path.realpath(__file__))
data_grouped = pd.read_csv(r''+S+"/corona virus/final_data.csv", parse_dates=['Date'])
data_grouped.drop('Unnamed: 0', axis=1, inplace=True)
data_grouped.State = data_grouped.State.fillna('NaN')
total_region = data_grouped.groupby(['Region', 'Date'], as_index=False)[
    'State', 'Confirmed', 'Recovered', 'Deaths', 'still infected'].sum()
test = total_region.groupby(['Region', 'Date'], as_index=False)[
    'Confirmed', 'Recovered', 'Deaths', 'still infected'].sum()
test = test.groupby('Date', as_index=False)['Confirmed', 'Recovered', 'Deaths', 'still infected'].sum()
for i in range(0, 46):
    total_region = total_region.append({'Region': 'Worldwide',
                                        'Date': test.Date.loc[i],
                                        'Confirmed': test.Confirmed.loc[i],
                                        'Recovered': test.Recovered.loc[i],
                                        'Deaths': test.Deaths.loc[i],
                                        'still infected': test['still infected'].loc[i]
                                        }, ignore_index=True)

table_info = data_grouped.groupby('Region')['Confirmed', 'Recovered', 'Deaths'].max()
table_info = table_info.sort_values(['Confirmed'], ascending=False)
table_info = table_info.reset_index()

# bar chart china vs the world :
china = data_grouped.groupby('Region')[
    'State', 'Lat', 'Long', 'Date', 'Confirmed', 'Recovered', 'Deaths', 'still infected'].get_group('China')
china_total = china[['State', 'Date', 'Confirmed', 'Recovered', 'Deaths', 'still infected']]
china_total = china_total.groupby(['State'], as_index=False)['Confirmed', 'Recovered', 'Deaths', 'still infected'].sum()
china_bar_confirmed = px.bar(china_total[['State', 'Confirmed']].sort_values('Confirmed', ascending=False),
                             x="Confirmed", y="State", color='State', orientation='h',
                             log_x=True, color_discrete_sequence=px.colors.qualitative.Bold, title='Confirmed Cases in '
                                                                                                   'China Per State',
                             height=800)

china_bar_deaths = px.bar(china_total[['State', 'Deaths']].sort_values('Deaths', ascending=False),
                          x="Deaths", y="State", color='State', orientation='h',
                          log_x=True, color_discrete_sequence=px.colors.qualitative.Bold, title='Deaths Cases in '
                                                                                                'China Per State',
                          height=800)

# world dataframe regions :
temp = data_grouped[['Region', 'Date', 'Confirmed', 'Deaths', 'Recovered']]
temp = temp.reset_index()

world_bar_confirmed = px.bar(temp, x="Date", y="Confirmed", color='Region', orientation='v', height=600,
                             title='Confirmed Cases In The World Per Region',
                             color_discrete_sequence=px.colors.cyclical.mygbm)

world_bar_deaths = px.bar(temp, x="Date", y="Deaths", color='Region', orientation='v', height=600,
                          title='Deaths Cases In The World Per Region',
                          color_discrete_sequence=px.colors.cyclical.mygbm)

# worldwide spread map :
formated_gdf = data_grouped.groupby(['Date', 'Region'])['Confirmed', 'Deaths', 'Recovered'].max()
formated_gdf = formated_gdf.reset_index()
formated_gdf['Date'] = formated_gdf['Date'].dt.strftime('%m/%d/%Y')
formated_gdf['size'] = formated_gdf['Confirmed'].pow(0.5)

world_spread = px.scatter_geo(formated_gdf,
                              locations="Region", locationmode='country names',
                              color="Confirmed", size='size', hover_name="Region",
                              range_color=[0, max(formated_gdf['Confirmed']) + 2],
                              projection="natural earth", animation_frame="Date",
                              title='Spread over time')
world_spread.update(layout_coloraxis_showscale=False)
# china spread map:
china_grouped = china.groupby(['Date', 'State'])['Confirmed', 'Deaths', 'Recovered', 'Lat', 'Long'].max()
china_grouped = china_grouped.reset_index()
china_grouped['size'] = china_grouped.Confirmed.pow(0.5)
china_grouped['Date'] = china_grouped['Date'].dt.strftime('%m/%d/%Y')

china_spread = px.scatter_geo(china_grouped, lat='Lat', lon='Long', scope='asia',
                              color="size", size='size', hover_name='State',
                              hover_data=['Confirmed', 'Deaths', 'Recovered'],
                              projection="natural earth", animation_frame="Date",
                              title='Spread in China over time')
china_spread.update(layout_coloraxis_showscale=False)

# predictions :
# function to add daily cases that we need in predictions and the callback:


def daily_df(region):
    cum_df = total_region.groupby('Region', as_index=False)[
        'Date', 'Confirmed', 'Recovered', 'Deaths', 'still infected'].get_group(region)
    # adding daily columns :
    for k in cum_df.columns[1:]:
        diff = []
        test_list = []
        indexing = list(cum_df.index)
        for i, j in zip(indexing, indexing[1:-1]):
            diff.append(cum_df[str(k)].loc[j] - cum_df[str(k)].loc[i])
        test_list.append(cum_df[str(k)].tolist()[0])
        for c in diff:
            test_list.append(c)
        test_list.append(cum_df[str(k)].tolist()[-1] - cum_df[str(k)].tolist()[-2])
        cum_df['daily_' + str(k)] = test_list
    return cum_df

# worldwide cumulative predictions functions  :
def predict_ww(status):
    cum_df = daily_df('Worldwide')
    if status in cum_df.columns:
        df = cum_df.groupby('Date',as_index=False)[status].sum()
        df.reset_index(drop=True,inplace=True)
        df.rename(columns={"Date": "ds", status: "y"},inplace=True)
        m = Prophet(yearly_seasonality=False,
                    weekly_seasonality = False,
                    daily_seasonality = False,
                    seasonality_mode = 'additive')
        m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        m.add_seasonality(name='weekly', period=7, fourier_order=21)
        m.add_seasonality(name='daily', period=1, fourier_order=3)
        m.fit(df)
        future_dates = m.make_future_dataframe(periods=15)
        prediction = m.predict(future_dates)
        forecast = pd.DataFrame()
        forecast['Date'] = prediction['ds']
        forecast['Future '+status] = prediction['yhat']
        return forecast,cum_df


forecast_cum_confirmed_ww = predict_ww('Confirmed')[0][-16:]
forecast_cum_deaths_ww = predict_ww('Deaths')[0][-16:]
forecast_cum_Recovered_ww = predict_ww('Recovered')[0][-16:]

# predict daily worldwide :
def predict_daily_ww(status):
    cum_df = daily_df('Worldwide')
    if status in cum_df.columns:
        df = cum_df.groupby('Date',as_index=False)[status].sum()
        df.reset_index(drop=True,inplace=True)
        df.rename(columns={"Date": "ds", status: "y"},inplace=True)
        m = Prophet()
        m.fit(df)
        future_dates = m.make_future_dataframe(periods=15)
        prediction = m.predict(future_dates)
        forecast = pd.DataFrame()
        forecast['Date'] = prediction['ds']
        forecast['Future '+status] = prediction['yhat']
        return forecast,cum_df


forecast_daily_confirmed_ww = predict_daily_ww('daily_Confirmed')[0][-16:]
forecast_daily_deaths_ww = predict_daily_ww('daily_Deaths')[0][-16:]
forecast_daily_Recovered_ww = predict_daily_ww('daily_Recovered')[0][-16:]
# adding dataframes to figure (wordlwide):


cum_df = daily_df('Worldwide')
confirmed_fig_ww = go.Figure({
        'data': [
            {
                'type': 'scatter',
                'x': cum_df.Date,
                'y': cum_df.daily_Confirmed,
                'mode': 'lines+markers',
                'marker': {'color': '#036bfc'},
                'name': 'Daily Confirmed Cases',
                'hoverlabel': {'namelength': 25}
            },
            {
                'type': 'scatter',
                'x': cum_df.Date,
                'y': cum_df.Confirmed,
                'mode': 'lines+markers',
                'marker': {'color': '#F4B000'},
                'name': 'Cumulative Confirmed Cases',
                'hoverlabel': {'namelength': 25}
            },
            {
                'type': 'scatter',
                'x': forecast_cum_confirmed_ww.Date,
                'y': forecast_cum_confirmed_ww['Future Confirmed'],
                'mode': 'lines',
                'marker': {'color': 'black'},
                'name': 'Future Confirmed Cases',
                'hoverlabel': {'namelength': 25}
            },
            {
                'type': 'scatter',
                'x': forecast_daily_confirmed_ww.Date,
                'y': forecast_daily_confirmed_ww['Future daily_Confirmed'],
                'mode': 'lines',
                'marker': {'color': 'black'},
                'name': 'Future Confirmed Cases',
                'hoverlabel': {'namelength': 25}
            }
        ],
        'layout': {
            'autosize': True,
            'legend': {'bgcolor': 'rgba(255,255,255,0)', 'x': 0, 'y': 1},
            'xaxis': {'tickformat': '%m-%d', 'title': {'text': "Date"}},
            'margin': {"r": 0, "t": 10, "l": 60, "b": 50},
            'plot_bgcolor': 'rgba(255,255,255,1)',
            'paper_bgcolor': 'rgba(255,255,255,1)',
            'hovermode': 'x',
            'dragmode': False
        }
    })
recovered_fig_ww = go.Figure({
    'data': [
        {
            'type': 'scatter',
            'x': cum_df.Date,
            'y': cum_df.daily_Recovered,
            'mode': 'lines+markers',
            'marker': {'color': '#036bfc'},
            'name': 'Daily Recovered Cases',
            'hoverlabel': {'namelength': 25},
        },
        {
            'type': 'scatter',
            'x': cum_df.Date,
            'y': cum_df.Recovered,
            'mode': 'lines+markers',
            'marker': {'color': '#28A745'},
            'name': 'Cumulative Recovered Cases',
            'hoverlabel': {'namelength': 25}
        },
        {
            'type': 'scatter',
            'x': forecast_daily_Recovered_ww.Date,
            'y': forecast_daily_Recovered_ww['Future daily_Recovered'],
            'mode': 'lines',
            'marker': {'color': 'black'},
            'name': 'Future Recovered Cases',
            'hoverlabel': {'namelength': 25},
        },
        {
            'type': 'scatter',
            'x': forecast_cum_Recovered_ww.Date,
            'y': forecast_cum_Recovered_ww['Future Recovered'],
            'mode': 'lines',
            'marker': {'color': 'black'},
            'name': 'Future Recovered Cases',
            'hoverlabel': {'namelength': 25},
        }
    ],
    'layout': {
        'autosize': True,
        'legend': {'bgcolor': 'rgba(255,255,255,0)', 'x': 0, 'y': 1},
        'xaxis': {'tickformat': '%m-%d', 'title': {'text': "Date"}},
        'margin': {"r": 0, "t": 10, "l": 60, "b": 50},
        'plot_bgcolor': 'rgba(255,255,255,1)',
        'paper_bgcolor': 'rgba(255,255,255,1)',
        'hovermode': 'x',
        'dragmode': False
    }
})
deaths_fig_ww = go.Figure({
    'data': [
        {
            'type': 'scatter',
            'x': cum_df.Date,
            'y': cum_df.daily_Deaths,
            'mode': 'lines+markers',
            'marker': {'color': '#036bfc'},
            'name': 'Daily Deaths Cases',
            'hoverlabel': {'namelength': 25},
        },
        {
            'type': 'scatter',
            'x': cum_df.Date,
            'y': cum_df.Deaths,
            'mode': 'lines+markers',
            'marker': {'color': '#E55465'},
            'name': 'Cumulative Deaths Cases',
            'hoverlabel': {'namelength': 25}
        },
        {
            'type': 'scatter',
            'x': forecast_daily_deaths_ww.Date,
            'y': forecast_daily_deaths_ww['Future daily_Deaths'],
            'mode': 'lines',
            'marker': {'color': 'black'},
            'name': 'Future Deaths Cases',
            'hoverlabel': {'namelength': 25},
        },
        {
            'type': 'scatter',
            'x': forecast_cum_deaths_ww.Date,
            'y': forecast_cum_deaths_ww['Future Deaths'],
            'mode': 'lines',
            'marker': {'color': 'black'},
            'name': 'Future Deaths Cases',
            'hoverlabel': {'namelength': 25},
        }
    ],
    'layout': {
        'autosize': True,
        'legend': {'bgcolor': 'rgba(255,255,255,0)', 'x': 0, 'y': 1},
        'xaxis': {'tickformat': '%m-%d', 'title': {'text': "Date"}},
        'margin': {"r": 0, "t": 10, "l": 60, "b": 50},
        'plot_bgcolor': 'rgba(255,255,255,1)',
        'paper_bgcolor': 'rgba(255,255,255,1)',
        'hovermode': 'x',
        'dragmode': False
    }
})

# function to predict for China :
# dataframe cum_df for china


def predict_china(status):
    cum_df = daily_df('China')
    df = cum_df.groupby('Date',as_index=False)[status].sum()
    df = df[['Date',status]]
    df.rename(columns={"Date": "ds", status: "y"},inplace=True)
    df.reset_index(drop=True,inplace=True)
    m = Prophet( yearly_seasonality=False,
                weekly_seasonality = False,
                daily_seasonality = False,
                seasonality_mode = 'additive')
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    m.add_seasonality(name='weekly', period=7, fourier_order=21)
    m.add_seasonality(name='daily', period=1, fourier_order=3)
    m.fit(df)
    future_dates = m.make_future_dataframe(periods=15)
    prediction = m.predict(future_dates)
    forecast = pd.DataFrame()
    forecast['Date'] = prediction['ds']
    forecast['Future ' + status] = prediction['yhat']
    return forecast,cum_df


forecast_cum_confirmed_china = predict_china('Confirmed')[0][-16:]
forecast_cum_deaths_china = predict_china('Deaths')[0][-16:]
forecast_cum_Recovered_china = predict_china('Recovered')[0][-16:]

# predict daily for china


def predict_china_daily(status):
    cum_df = daily_df('China')
    df = cum_df.groupby('Date', as_index=False)[status].sum()
    df = df[['Date', status]]
    df.rename(columns={"Date": "ds", status: "y"}, inplace=True)
    df.reset_index(drop=True,inplace=True)
    m = Prophet()
    m.fit(df)
    future_dates = m.make_future_dataframe(periods=15)
    prediction = m.predict(future_dates)
    forecast = pd.DataFrame()
    forecast['Date'] = prediction['ds']
    forecast['Future ' + status] = prediction['yhat']
    return forecast,cum_df


forecast_daily_confirmed_china = predict_china_daily('daily_Confirmed')[0][-16:]
forecast_daily_deaths_china = predict_china_daily('daily_Deaths')[0][-16:]
forecast_daily_Recovered_china = predict_china_daily('daily_Recovered')[0][-16:]

# adding figures for china :

cum_df = daily_df('China')
confirmed_fig_china = go.Figure({
        'data': [
            {
                'type': 'scatter',
                'x': cum_df.Date,
                'y': cum_df.daily_Confirmed,
                'mode': 'lines+markers',
                'marker': {'color': '#036bfc'},
                'name': 'Daily Confirmed Cases',
                'hoverlabel': {'namelength': 25}
            },
            {
                'type': 'scatter',
                'x': cum_df.Date,
                'y': cum_df.Confirmed,
                'mode': 'lines+markers',
                'marker': {'color': '#F4B000'},
                'name': 'Cumulative Confirmed Cases',
                'hoverlabel': {'namelength': 25}
            },
            {
                'type': 'scatter',
                'x': forecast_cum_confirmed_china.Date,
                'y': forecast_cum_confirmed_china['Future Confirmed'],
                'mode': 'lines',
                'marker': {'color': 'black'},
                'name': 'Future Confirmed Cases',
                'hoverlabel': {'namelength': 25}
            },
            {
                'type': 'scatter',
                'x': forecast_daily_confirmed_china.Date,
                'y': forecast_daily_confirmed_china['Future daily_Confirmed'],
                'mode': 'lines',
                'marker': {'color': 'black'},
                'name': 'Future Confirmed Cases',
                'hoverlabel': {'namelength': 25}
            }
        ],
        'layout': {
            'autosize': True,
            'legend': {'bgcolor': 'rgba(255,255,255,0)', 'x': 0, 'y': 1},
            'xaxis': {'tickformat': '%m-%d', 'title': {'text': "Date"}},
            'margin': {"r": 0, "t": 10, "l": 60, "b": 50},
            'plot_bgcolor': 'rgba(255,255,255,1)',
            'paper_bgcolor': 'rgba(255,255,255,1)',
            'hovermode': 'x',
            'dragmode': False
        }
    })
recovered_fig_china = go.Figure({
    'data': [
        {
            'type': 'scatter',
            'x': cum_df.Date,
            'y': cum_df.daily_Recovered,
            'mode': 'lines+markers',
            'marker': {'color': '#036bfc'},
            'name': 'Daily Recovered Cases',
            'hoverlabel': {'namelength': 25},
        },
        {
            'type': 'scatter',
            'x': cum_df.Date,
            'y': cum_df.Recovered,
            'mode': 'lines+markers',
            'marker': {'color': '#28A745'},
            'name': 'Cumulative Recovered Cases',
            'hoverlabel': {'namelength': 25}
        },
        {
            'type': 'scatter',
            'x': forecast_daily_Recovered_china.Date,
            'y': forecast_daily_Recovered_china['Future daily_Recovered'],
            'mode': 'lines',
            'marker': {'color': 'black'},
            'name': 'Future Recovered Cases',
            'hoverlabel': {'namelength': 25},
        },
        {
            'type': 'scatter',
            'x': forecast_cum_Recovered_china.Date,
            'y': forecast_cum_Recovered_china['Future Recovered'],
            'mode': 'lines',
            'marker': {'color': 'black'},
            'name': 'Future Recovered Cases',
            'hoverlabel': {'namelength': 25},
        }
    ],
    'layout': {
        'autosize': True,
        'legend': {'bgcolor': 'rgba(255,255,255,0)', 'x': 0, 'y': 1},
        'xaxis': {'tickformat': '%m-%d', 'title': {'text': "Date"}},
        'margin': {"r": 0, "t": 10, "l": 60, "b": 50},
        'plot_bgcolor': 'rgba(255,255,255,1)',
        'paper_bgcolor': 'rgba(255,255,255,1)',
        'hovermode': 'x',
        'dragmode': False
    }
})
deaths_fig_china = go.Figure({
    'data': [
        {
            'type': 'scatter',
            'x': cum_df.Date,
            'y': cum_df.daily_Deaths,
            'mode': 'lines+markers',
            'marker': {'color': '#036bfc'},
            'name': 'Daily Deaths Cases',
            'hoverlabel': {'namelength': 25},
        },
        {
            'type': 'scatter',
            'x': cum_df.Date,
            'y': cum_df.Deaths,
            'mode': 'lines+markers',
            'marker': {'color': '#E55465'},
            'name': 'Cumulative Deaths Cases',
            'hoverlabel': {'namelength': 25}
        },
        {
            'type': 'scatter',
            'x': forecast_daily_deaths_china.Date,
            'y': forecast_daily_deaths_china['Future daily_Deaths'],
            'mode': 'lines',
            'marker': {'color': 'black'},
            'name': 'Future Deaths Cases',
            'hoverlabel': {'namelength': 25},
        },
        {
            'type': 'scatter',
            'x': forecast_cum_deaths_china.Date,
            'y': forecast_cum_deaths_china['Future Deaths'],
            'mode': 'lines',
            'marker': {'color': 'black'},
            'name': 'Future Deaths Cases',
            'hoverlabel': {'namelength': 25},
        }
    ],
    'layout': {
        'autosize': True,
        'legend': {'bgcolor': 'rgba(255,255,255,0)', 'x': 0, 'y': 1},
        'xaxis': {'tickformat': '%m-%d', 'title': {'text': "Date"}},
        'margin': {"r": 0, "t": 10, "l": 60, "b": 50},
        'plot_bgcolor': 'rgba(255,255,255,1)',
        'paper_bgcolor': 'rgba(255,255,255,1)',
        'hovermode': 'x',
        'dragmode': False
    }
})

# app layout

app.layout = html.Div([
    html.Div(
        html.H2(children=["Corona Virus Analysis Dashboard 2019"], style={'text-align': 'center'})
    ),
    html.Div(
        [
            html.Div(
                [
                    html.H6(
                        ['Filter menus'],
                    ),
                    html.Div([
                        html.Div([
                            dcc.Dropdown(
                                id='dropdown_menu',
                                options=[
                                    {'label': i, 'value': i} for i in list(total_region.Region.drop_duplicates())
                                ],
                                # searchable= False,
                                clearable=False,
                                value='Worldwide'
                            )
                        ], className='col-md-8'),
                        html.Br(),
                        html.Br(),
                        html.Div([
                            dcc.Dropdown(
                                id='status',
                                options=[
                                    {'label': i, 'value': i} for i in total_region.columns[2:].tolist()
                                ],
                                value='Confirmed',
                                clearable=False)
                        ], className='col-md-8')
                    ], className='row'),
                    html.Br(),
                    html.H6(
                        ['Table info'],
                    ),
                    html.Br(),
                    dt.DataTable(
                        data=table_info.to_dict('records'),
                        columns=[{'id': c, 'name': c} for c in table_info.columns.tolist()],
                        sort_action='native',
                        fixed_rows={"headers": True},
                        sort_mode="multi",
                        # style_as_list_view=True,
                        style_table={
                            'minWidth': '100%',
                            "height": "65vh",
                        },
                        style_header={
                            "height": "2vw",
                            'minWidth': '100%'
                        },
                        style_cell={
                            'textAlign': 'left',
                            'color': 'black',
                            'fontSize': 14,
                            'fontFamily': 'sans-serif',
                        },
                        css=[{'selector': '.row', 'rule': 'margin: 0'}],
                        style_cell_conditional=[
                            {
                                "if": {"column_id": "Region", },
                                "minWidth": "4vw",
                                "width": "4vw",
                                "maxWidth": "10vw",
                            },
                            {
                                "if": {"column_id": "Confirmed", },
                                "color": "#F4B000"
                            },
                            {
                                "if": {"column_id": "Recovered", },
                                "color": "#6EB24D"
                            },
                            {
                                "if": {"column_id": "Deaths", },
                                "color": "#E55465"
                            }
                        ]
                    )
                ],
                className='grid-item', id='infobox_container'
            ),
            html.Div(
                [
                    html.Div(id='info-container'),
                    html.Div(
                        [
                            html.H6(
                                ['Map'],
                            ),
                            dcc.Graph(
                                id='mapbox',
                                config={
                                    'modeBarButtonsToRemove': [
                                        'select2d',
                                        'lasso2d',
                                        'hoverClosestGeo',
                                    ],
                                    'scrollZoom': True
                                },
                                style={'position': 'relative', 'width': '100%', 'height': '90%', 'left': '0px',
                                       'top': '0px'}
                            ),
                        ],
                        className='grid-item',
                        style={'width': '100%',
                               'position': 'relative'}
                    )
                ])
        ], className='grid-container-onethird-twothirds-cols'),
    html.Div(
        [html.Div([
            html.H6('Chart Of The Total Recorded Rases'),
            html.Div([
                dcc.Graph(
                id='treemap',
                className='col-md-12'
            )
        ], className='row')]
            , className='grid-item')]
        , className='grid-container-one-col'),
    html.Br(),
    html.Div([
        html.Div([
            html.H6(id='confirmed_title'),
            html.Br(),
            dcc.Graph(id='confirmed-fig')
        ], className='grid-item'),
        html.Div([
            html.H6(id='recovered_title'),
            html.Br(),
            dcc.Graph(id='recovered-fig')
        ], className='grid-item')
    ], className='grid-container-two-cols'),
    html.Div([
        html.Div([
            html.H6(id='deaths_title'),
            html.Br(),
            dcc.Graph(id='deaths-fig')
        ], className='grid-item'),
        html.Div([
            html.H6(id='active_title'),
            html.Br(),
            dcc.Graph(id='still-infected-fig')
        ], className='grid-item')
    ], className='grid-container-two-cols'),
    html.Div([
        html.H3('Special China Analysis, Comparing with Worldwide spread')
    ]),
    html.Div([
        html.Div([
            dcc.Graph(
                figure=china_bar_confirmed
            )
        ], className='grid-item'),
    ], className='grid-container-one-col'),
    html.Br(),
    html.Div([
        html.Div([
            dcc.Graph(
                figure=world_bar_confirmed
            )
        ], className='grid-item')
    ], className='grid-container-one-col'),
    html.Br(),
    html.Div([
        html.Div([
            dcc.Graph(
                figure=china_bar_deaths
            )
        ], className='grid-item')
    ], className='grid-container-one-col'),
    html.Br(),
    html.Div([
        html.Div([
            dcc.Graph(
                figure=world_bar_deaths
            )
        ], className='grid-item')
    ], className='grid-container-one-col'),
    html.Br(),
    html.Div([
        html.Div([
            html.H6('Covid-19 Spread Over Time In The World'),
            dcc.Graph(
                figure=world_spread
            )
        ], className='grid-item'),
        html.Div([
            html.H6('Covid-19 Spread Over Time In China'),
            dcc.Graph(
                figure=china_spread
            )
        ], className='grid-item')
    ], className='grid-container-two-cols'),
    html.H3('Future Predictions For The World'),
    html.Div([
        html.Div([
            dcc.Graph(
                figure=confirmed_fig_ww
            )
        ],className='grid-item'),
        html.Div([
            dcc.Graph(
                figure=recovered_fig_ww
            )
        ],className='grid-item'),
        html.Div([
            dcc.Graph(
                figure=deaths_fig_ww
            )
        ],className='grid-item'),
    ],className='grid-container-three-cols'),
    html.H3('Future Predictions for China'),
    html.Div([
        html.Div([
            dcc.Graph(
                figure=confirmed_fig_china
            )
        ],className='grid-item'),
        html.Div([
            dcc.Graph(
                figure=recovered_fig_china
            )
        ],className='grid-item'),
        html.Div([
            dcc.Graph(
                figure=deaths_fig_china
            )
        ],className='grid-item'),
    ],className='grid-container-three-cols')
], id='main-container')


# mapbox function :
def mapBox(region, status):
    if str(status) == 'Recovered':
        color_list = ['#6EB24D', '#6ABF40', '#66CC33', '#62D926', '#5DE619']
    else:
        color_list = ["#FA9090", "#F77A7A", "#F56666", "#F15454", "#ED4343"]
    map_access_token = 'pk.eyJ1Ijoib3Vzc2FtYWxoIiwiYSI6ImNrOXE5YjhyeTBpMXMzbm1tNzl6c3YycG8ifQ.ItAzwdmH4IXBfK8op6rptg'
    px.set_mapbox_access_token(map_access_token)
    # df['Date'] = df['Date'].dt.strftime('%m/%d/%Y')
    if str(region) == 'Worldwide':
        df = data_grouped.groupby(['Region', 'State', 'Date'])['Confirmed', 'Recovered', 'Deaths', 'Lat', 'Long'].max()
        df = df.reset_index()
        df['size'] = df[str(status)].pow(0.5)
        fig = px.scatter_mapbox(df,
                                lat="Lat",
                                lon="Long",
                                color=str(status),
                                size="size",
                                zoom=3,
                                color_continuous_scale=color_list,
                                hover_data=[str(status)],
                                size_max=40)
        fig.update_layout({
            'margin': {"r": 0, "t": 0, "l": 0, "b": 0}
        })
        return fig
    else:
        df = data_grouped.groupby('Region', as_index=False)['Date', 'State', 'Confirmed', 'Deaths',
                                                            'Recovered', 'Lat', 'Long'].get_group(str(region))
        df = df.groupby('State')['Date', 'Confirmed', 'Deaths', 'Recovered', 'Lat', 'Long'].max()
        df = df.reset_index()
        df['size'] = df[str(status)].pow(0.3)
        fig = px.scatter_mapbox(df,
                                lat="Lat",
                                lon="Long",
                                color=str(status),
                                size="size",
                                zoom=3,
                                color_continuous_scale=color_list,
                                hover_data=[str(status)],
                                size_max=40)
        fig.update_layout({
            'margin': {"r": 0, "t": 0, "l": 0, "b": 0}

        })

        return fig


@app.callback(
    [Output('info-container', 'children'),
     Output('mapbox', 'figure'),
     Output('treemap', 'figure'),
     Output('confirmed-fig', 'figure'),
     Output('recovered-fig', 'figure'),
     Output('deaths-fig', 'figure'),
     Output('still-infected-fig', 'figure'),
     Output('confirmed_title', 'children'),
     Output('recovered_title', 'children'),
     Output('deaths_title', 'children'),
     Output('active_title', 'children')],
    [Input('dropdown_menu', 'value'),
     Input('status', 'value')])
def update_figure(region, status):
    test_df = total_region.groupby('Region', as_index=False)[
        'Confirmed', 'Deaths', 'Recovered', 'still infected'].get_group(str(region))
    confirmed = test_df['Confirmed'].max()
    recovered = test_df['Recovered'].max()
    deaths = test_df['Deaths'].max()
    active = test_df['still infected'].max()
    ## treemap
    temp = total_region.groupby('Region', as_index=False)[
        'Date', 'Confirmed', 'Deaths', 'Recovered', 'still infected'].get_group(str(region))
    temp = temp.groupby('Date', as_index=False)['Confirmed', 'Deaths', 'Recovered', 'still infected'].sum()
    temp = temp.sort_values('Date', ascending=False)
    temp.reset_index(drop=True, inplace=True)
    # colors :
    d = '#ff2e63'
    r = '#30e3ca'
    i = '#f8b400'
    ############################
    tm = temp.head(1).melt(id_vars="Date", value_vars=['still infected', 'Deaths', 'Recovered'])
    tm['Confirmed'] = 'Confirmed'
    treemap_figure = px.treemap(tm, path=["Confirmed", "variable"], values="value")
    treemap_figure.update_layout(
        height=200
    )

    ## dataframe for line chart
    cum_df = total_region.groupby('Region', as_index=False)[
        'Date', 'Confirmed', 'Recovered', 'Deaths', 'still infected'].get_group(region)
    ## adding daily columns :
    for k in cum_df.columns[1:]:
        diff = []
        test_list = []
        indexing = list(cum_df.index)
        for i, j in zip(indexing, indexing[1:-1]):
            diff.append(cum_df[str(k)].loc[j] - cum_df[str(k)].loc[i])
        test_list.append(cum_df[str(k)].tolist()[0])
        for c in diff:
            test_list.append(c)
        test_list.append(cum_df[str(k)].tolist()[-1] - cum_df[str(k)].tolist()[-2])
        cum_df['daily_' + str(k)] = test_list
    ## line charts
    confirmed_fig = go.Figure({
        'data': [
            {
                'type': 'scatter',
                'x': cum_df.Date,
                'y': cum_df.daily_Confirmed,
                'mode': 'lines+markers',
                'marker': {'color': '#036bfc'},
                'name': 'Daily Confirmed Cases',
                'hoverlabel': {'namelength': 25}
            },
            {
                'type': 'scatter',
                'x': cum_df.Date,
                'y': cum_df.Confirmed,
                'mode': 'lines+markers',
                'marker': {'color': '#F4B000'},
                'name': 'Cumulative Confirmed Cases',
                'hoverlabel': {'namelength': 25}
            }
        ],
        'layout': {
            'autosize': True,
            'legend': {'bgcolor': 'rgba(255,255,255,0)', 'x': 0, 'y': 1},
            'xaxis': {'tickformat': '%m-%d', 'title': {'text': "Date"}},
            'yaxis': {'title': {'text': 'Confirmed Cases'},
                      'gridcolor': '#f5f5f5'
                      },
            'margin': {"r": 0, "t": 10, "l": 60, "b": 50},
            'plot_bgcolor': 'rgba(255,255,255,1)',
            'paper_bgcolor': 'rgba(255,255,255,1)',
            'hovermode': 'x',
            'dragmode': False
        }
    })
    recovered_fig = go.Figure({
        'data': [
            {
                'type': 'scatter',
                'x': cum_df.Date,
                'y': cum_df.daily_Recovered,
                'mode': 'lines+markers',
                'marker': {'color': '#036bfc'},
                'name': 'Daily Recovered Cases',
                'hoverlabel': {'namelength': 25},
            },
            {
                'type': 'scatter',
                'x': cum_df.Date,
                'y': cum_df.Recovered,
                'mode': 'lines+markers',
                'marker': {'color': '#28A745'},
                'name': 'Cumulative Recovered Cases',
                'hoverlabel': {'namelength': 25}
            }
        ],
        'layout': {
            'autosize': True,
            'legend': {'bgcolor': 'rgba(255,255,255,0)', 'x': 0, 'y': 1},
            'xaxis': {'tickformat': '%m-%d', 'title': {'text': "Date"}},
            'yaxis': {'title': {'text': 'Recovered Cases'},
                      'gridcolor': '#f5f5f5'
                      },
            'margin': {"r": 0, "t": 10, "l": 60, "b": 50},
            'plot_bgcolor': 'rgba(255,255,255,1)',
            'paper_bgcolor': 'rgba(255,255,255,1)',
            'hovermode': 'x',
            'dragmode': False
        }
    })
    deaths_fig = go.Figure({
        'data': [
            {
                'type': 'scatter',
                'x': cum_df.Date,
                'y': cum_df.daily_Deaths,
                'mode': 'lines+markers',
                'marker': {'color': '#036bfc'},
                'name': 'Daily Deaths Cases',
                'hoverlabel': {'namelength': 25},
            },
            {
                'type': 'scatter',
                'x': cum_df.Date,
                'y': cum_df.Deaths,
                'mode': 'lines+markers',
                'marker': {'color': '#E55465'},
                'name': 'Cumulative Deaths Cases',
                'hoverlabel': {'namelength': 25}
            }
        ],
        'layout': {
            'autosize': True,
            'legend': {'bgcolor': 'rgba(255,255,255,0)', 'x': 0, 'y': 1},
            'xaxis': {'tickformat': '%m-%d', 'title': {'text': "Date"}},
            'yaxis': {'title': {'text': 'Deaths Cases'},
                      'gridcolor': '#f5f5f5'
                      },
            'margin': {"r": 0, "t": 10, "l": 60, "b": 50},
            'plot_bgcolor': 'rgba(255,255,255,1)',
            'paper_bgcolor': 'rgba(255,255,255,1)',
            'hovermode': 'x',
            'dragmode': False
        }
    })
    still_infected_fig = go.Figure({
        'data': [
            {
                'type': 'scatter',
                'x': cum_df.Date,
                'y': cum_df['still infected'],
                'mode': 'lines+markers',
                'marker': {'color': '#DD1E34'},
                'name': 'Cumulative Active Cases',
                'hoverlabel': {'namelength': 25}
            }
        ],
        'layout': {
            'autosize': True,
            'legend': {'bgcolor': 'rgba(255,255,255,0)', 'x': 0, 'y': 1},
            'xaxis': {'tickformat': '%m-%d', 'title': {'text': "Date"}},
            'yaxis': {'title': {'text': 'Active Cases'},
                      'gridcolor': '#f5f5f5'
                      },
            'margin': {"r": 0, "t": 10, "l": 60, "b": 50},
            'plot_bgcolor': 'rgba(255,255,255,1)',
            'paper_bgcolor': 'rgba(255,255,255,1)',
            'hovermode': 'x',
            'dragmode': False
        }
    })
    return ([
                html.Div(
                    [html.H3(confirmed, id='confirmed_text',
                             style={'color': '#F4B000'}),
                     html.P(['Confirmed'], id='confirmed_label',
                            style={'color': '#F4B000'})],
                    id='confirmed_div',
                    className='mini_container',
                ),
                html.Div(
                    [html.H3(recovered, id='recovered_text',
                             style={'color': '#28A745'}),
                     html.P(['Recovered'], id='recovered_label',
                            style={'color': '#28A745'})],
                    id='recovered_div',
                    className='mini_container',
                ),
                html.Div(
                    [html.H3(deaths, id='deaths_text',
                             style={'color': '#E55465'}),
                     html.P(['Deaths'], id='deaths_label',
                            style={'color': '#E55465'})],
                    id='deaths_div',
                    className='mini_container',
                ),
                html.Div(
                    [html.H3(active, id='active_text',
                             style={'color': '#DD1E34'}),
                     html.P(['Active'], id='active_label',
                            style={'color': '#DD1E34'})],
                    id='active_div',
                    className='mini_container',
                )
            ], mapBox(region, status), treemap_figure, confirmed_fig, recovered_fig, deaths_fig, still_infected_fig,
            'Confirmed Cases in ' + region, 'Recovered Cases in ' + region, 'Deaths Cases in ' + region,
            'Active Cases in ' + region)


if __name__ == '__main__':
    app.run_server(debug=True)
