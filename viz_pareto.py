import os
import base64
from datetime import date
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dash
import flask
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_extendable_graph as deg
import plotly.graph_objects as go
import dash_html_components as html
import plotly.express as px
import plotly.io as pio
from covid_xprize.nixtamalai.viz_components import get_pareto_data
from covid_xprize.nixtamalai.viz_components import npi_val_to_cost
from covid_xprize.nixtamalai.viz_components import get_overall_data
from covid_xprize.nixtamalai.viz_components import npi_cost_to_val
from covid_xprize.nixtamalai.viz_components import get_sliders
import palettable as pltt
import dash_table

# TEMPLATE = 'plotly_dark'


START_DATE = "2020-08-01"
END_DATE = "2020-09-01"
TEST_COST = "covid_xprize/nixtamalai/viz_data/uniform_random_costs.csv"
INITIAL_COUNTRY = "Mexico"
IP_FILE = "covid_xprize/nixtamalai/viz_data/scenario_all_countries_no_regions.csv"
DEFAULT_COLORS = px.colors.qualitative.Plotly
logo_filename = "./covid_xprize/nixtamalai/img/logo.jpeg"
encoded_logo = base64.b64encode(open(logo_filename, 'rb').read())
HIST_DF = pd.read_csv(IP_FILE,
                        parse_dates=['Date'],
                        encoding="ISO-8859-1",
                        keep_default_na=False,
                        error_bad_lines=True)
HIST_DF = HIST_DF.replace("", np.NaN)
ALL_COUNTRIES = [{"label":c, "value":c} for c in HIST_DF.CountryName.unique()] 
WEIGHTS_DF = pd.read_csv(TEST_COST, keep_default_na=False)
WEIGHTS_DF = WEIGHTS_DF.replace("", np.NaN)

overall_pdf, predictions = get_overall_data(START_DATE, END_DATE, HIST_DF, WEIGHTS_DF,
                                            INITIAL_COUNTRY, "greedy")
# Gráfica inicial de Pareto
pareto = get_pareto_data(list(overall_pdf['Stringency']),
                         list(overall_pdf['PredictedDailyNewCases']))
pareto_data = {"x": pareto[0],
               "y": pareto[1],
               "name": "Base (Blind Greedy for Mexico)",
               "showlegend": True,
               }
npis = (WEIGHTS_DF
        .drop(columns=['CountryName', 'RegionName'])
        .to_dict(orient='records'))[0]
BASE_COSTS = npi_val_to_cost(npis)
# Gráfica inicial de radar
radar_data = {
    "r": [v for _,v in BASE_COSTS.items()],
    "theta": [k.split("_")[0] for k,_ in BASE_COSTS.items()],
    "name": "Base (Blind Greedy for Mexico)",
    'type': 'scatterpolar',
    "showlegend": True,
}

# Gráfica inicial  de predicciones
predictions = pd.concat(predictions)
predictions['Prescriptor'] = 0

fig = px.line(predictions,
    facet_col="Prescriptor",
    color="Prescriptor",
    line_group="PrescriptionIndex",
    x="Date",
    y="PredictedDailyNewCases",
    facet_col_wrap=3)

data_table = dash_table.DataTable(
    id='table',
    columns=[{"name": i, "id": i} for i in predictions.columns],
    data=predictions.to_dict('records'),
    data_previous=[dict()],
    export_format='xlsx',
    export_headers='display',
    page_size=10,
    sort_action='native'
)

sliders = get_sliders(BASE_COSTS)

server = flask.Flask(__name__) # define flask app.server
app = dash.Dash(__name__,
    external_stylesheets=[dbc.themes.FLATLY],
    prevent_initial_callbacks=True,
    server=server)

app.layout = dbc.Container(
    [
        dbc.Row(
            [dbc.Col(html.Img(src='data:image/png;base64,{}'.format(encoded_logo.decode()),
                     height="100px",style={'padding-left': '30px'}), width=1),
            dbc.Col(html.Div(html.H1(children='Visualizing Intervention Plans')))]
            ),
        dbc.Row(html.Hr()),
        dbc.Row(
            [
                dbc.Col(html.Div(sliders[0:3]), width=2),
                dbc.Col(html.Div(sliders[3:6]), width=2),
                dbc.Col(html.Div(sliders[6:9]), width=2),
                dbc.Col(html.Div(sliders[9:12]), width=2),
                dbc.Col(
                    [
                        html.Div(dcc.Dropdown(
                            id='model-selector',
                            options=[
                                {'label': 'Blind Greedy', 'value': 'greedy'},
                                {'label': 'Nixtamal Surrogate', 'value': 'nixtamal'}
                            ],
                            style={'color': 'black'},
                            value='greedy'
                        )),
                        html.Hr(),
                        html.Div(dcc.DatePickerRange(
                            id='date-range',
                            min_date_allowed=date(2020, 8, 1),
                            max_date_allowed=date(2021, 12, 31),
                            initial_visible_month=date(2020, 8, 1),
                            start_date=date(2020, 8, 1),
                            end_date=date(2020, 9, 1)
                        )),
                        html.Hr(),
                        html.Div(dcc.Dropdown(
                            id='country-selector',
                            options=ALL_COUNTRIES,
                            style={'color': 'black'},
                            value=INITIAL_COUNTRY
                        ))                                               
                    ],

                    width=2),
                dbc.Col(
                    [
                        html.Div(dbc.Button('Submit', id='submit-val',color="success",
                          n_clicks=0, block=True)),
                        html.Hr(),
                        html.Div(dbc.Button('Reset', id='reset-val', color="warning",
                        href='/', n_clicks=0, block=True))
                    ],

                    width=1),
            ], style={'padding-left': '30px'}
        ),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    html.Div(html.H4("NPI Weights"), style={
                             'text-align': 'center'}),
                    width={"size": 4, "offset": 1},
                ),
                dbc.Col(
                    html.Div(html.H4("Pareto Plot"), style={
                             'text-align': 'center'}),
                    width={"size": 6},
                ),
            ],
            align="center",
        ),
        dbc.Row(
            [
                dbc.Col(
                    deg.ExtendableGraph(
                            id='radar-plot',
                            figure=go.Figure(dict(
                                data=[radar_data],
                                layout={
                                        "legend": {"yanchor": "bottom", "y": 0.1, "x": -1.2},
                                        }                                
                            ))
                    ), width={"size": 4, "offset": 1},
                ),
                dbc.Col(
                    dcc.Loading(
                        id="loading-pareto",
                        children=[deg.ExtendableGraph(
                        id='pareto-plot',
                        figure=go.Figure(dict(
                            data=[pareto_data],
                            layout={
                                    "xaxis": {"title": "Mean Stringency"},
                                    "yaxis": {"title": "Mean New Cases per Day"},
                                    "legend": {"yanchor": "top", "y": 0.99, "x": 0.35},
                                    }
                        ))
                    )]), width={"size": 6},
                ),
            ],
            align="center",
        ),
        dbc.Row(dbc.Col(
            dcc.Loading(
                id="loadig-predictions",
                children=[dcc.Graph(id='predictions-graphs', figure=fig)]
            )

        )
        ),
        dbc.Row(dbc.Col(
            data_table, width="auto"),
        align='center', justify="center"),
    ], fluid=True
)

@app.callback([dash.dependencies.Output('pareto-plot', 'extendData'),
               dash.dependencies.Output('table', 'data_previous')],
               [dash.dependencies.Input('submit-val', 'n_clicks')],
               [dash.dependencies.State('C1-weight', 'value')],
               [dash.dependencies.State('C2-weight', 'value')],
               [dash.dependencies.State('C3-weight', 'value')],
               [dash.dependencies.State('C4-weight', 'value')],
               [dash.dependencies.State('C5-weight', 'value')],
               [dash.dependencies.State('C6-weight', 'value')],
               [dash.dependencies.State('C7-weight', 'value')],
               [dash.dependencies.State('C8-weight', 'value')],
               [dash.dependencies.State('H1-weight', 'value')],
               [dash.dependencies.State('H2-weight', 'value')],
               [dash.dependencies.State('H3-weight', 'value')],
               [dash.dependencies.State('H4-weight', 'value')],
               [dash.dependencies.State('model-selector', 'value')],
               [dash.dependencies.State('country-selector', 'value')],
               [dash.dependencies.State('date-range', 'start_date')],
               [dash.dependencies.State('date-range', 'end_date')],
               [dash.dependencies.State('pareto-plot', 'figure')]
            )

def update_pareto_plot(n_clicks, value_c1, value_c2, value_c3, value_c4, value_c5, value_c6,
               value_c7, value_c8, value_h1, value_h2, value_h3, value_h4, model, country, 
               start_date, end_date, figure):
    if n_clicks > 0:    
        weights_dict = {
            'CountryName': [country],
            'RegionName': [""],
            'C1_School closing': [value_c1],
            'C2_Workplace closing': [value_c2],
            'C3_Cancel public events': [value_c3],
            'C4_Restrictions on gatherings': [value_c4],
            'C5_Close public transport': [value_c5],
            'C6_Stay at home requirements': [value_c6],
            'C7_Restrictions on internal movement': [value_c7],
            'C8_International travel controls': [value_c8],
            'H1_Public information campaigns': [value_h1],
            'H2_Testing policy': [value_h2],
            'H3_Contact tracing': [value_h3],
            'H6_Facial Coverings': [value_h4]
        }
        prescriptor_names = {"greedy": "Blind Greedy",
                             "nixtamal": "Nixtamal Surrogate"}
        weights_dict = npi_cost_to_val(weights_dict)
        user_weights = pd.DataFrame.from_dict(weights_dict)
        overall_pdf, predictions = get_overall_data(
            start_date, end_date, HIST_DF, user_weights, country, model)
        pareto = get_pareto_data(list(overall_pdf['Stringency']),
                                 list(overall_pdf['PredictedDailyNewCases']))
        new_trace = {"x": pareto[0],
                     "y": pareto[1],
                    "name": "{} prescription {} for {}".format(prescriptor_names[model],
                                        n_clicks, country)
                    }
        predictions = pd.concat(predictions)
        predictions['Prescriptor'] = n_clicks
        prediction_traces = []
        for idx in predictions.PrescriptionIndex.unique():
            display_legend = True if idx == 0 else False
            idf = predictions[predictions.PrescriptionIndex == idx]
            trace = {"x": idf["Date"],
                     "y": idf["PredictedDailyNewCases"],
                     "mode": "lines",
                     "line": dict(color=DEFAULT_COLORS[n_clicks]),
                     "name": "{} prescription {} for {}".format(prescriptor_names[model],
                                                                n_clicks, country),
                     "legendgroup": "group_{}".format(n_clicks),
                     "showlegend": display_legend
                    }
            prediction_traces.append(trace)

        return ([new_trace, []], []), predictions.to_dict('records')
    return ([],[],[]), predictions.to_dict('records')

@app.callback(dash.dependencies.Output('radar-plot', 'extendData'),
               [dash.dependencies.Input('submit-val', 'n_clicks')],
               [dash.dependencies.State('C1-weight', 'value')],
               [dash.dependencies.State('C2-weight', 'value')],
               [dash.dependencies.State('C3-weight', 'value')],
               [dash.dependencies.State('C4-weight', 'value')],
               [dash.dependencies.State('C5-weight', 'value')],
               [dash.dependencies.State('C6-weight', 'value')],
               [dash.dependencies.State('C7-weight', 'value')],
               [dash.dependencies.State('C8-weight', 'value')],
               [dash.dependencies.State('H1-weight', 'value')],
               [dash.dependencies.State('H2-weight', 'value')],
               [dash.dependencies.State('H3-weight', 'value')],
               [dash.dependencies.State('H4-weight', 'value')],
               [dash.dependencies.State('model-selector', 'value')],
               [dash.dependencies.State('country-selector', 'value')],
               [dash.dependencies.State('radar-plot', 'figure')]
              )
def update_radar_plot(n_clicks, value_c1, value_c2, value_c3, value_c4, value_c5, value_c6,
               value_c7, value_c8, value_h1, value_h2, value_h3, value_h4, model, country, figure):
    if n_clicks > 0:    
        weights_dict = {
            'C1': value_c1,
            'C2': value_c2,
            'C3': value_c3,
            'C4': value_c4,
            'C5': value_c5,
            'C6': value_c6,
            'C7': value_c7,
            'C8': value_c8,
            'H1': value_h1,
            'H2': value_h2,
            'H3': value_h3,
            'H6': value_h4
        }
        prescriptor_names = {"greedy": "Blind Greedy",
                             "nixtamal": "Nixtamal Surrogate"}
        new_trace = {
            "r": [v for _,v in weights_dict.items()],
            "theta": [k for k,_ in weights_dict.items()],
            'type': 'scatterpolar',
            "name": "{} prescription {} for {}".format(prescriptor_names[model],
                                                                n_clicks, country),
        }
        return [new_trace, []], []

# @app.callback(
#     dash.dependencies.Output("pareto-plot", "figure"),
#     [dash.dependencies.Input("pareto-plot", "hoverData")],
#     [dash.dependencies.State('pareto-plot', 'figure')]
# )
# def highlight_trace(hover_data, figure):
#     # here you set the default settings
#     # for trace in my_pot.data:
#     #     country["line"]["width"] = 1
#     #     country["opacity"] = 0.5
#     if hover_data:
#         trace_index = hover_data["points"][0]["curveNumber"]
#         print(figure["data"])
#         # figure["data"][trace_index]["line"]["width"] = 5
#         # figure["data"][trace_index]["opacity"] = 1
#     return figure


@app.callback(dash.dependencies.Output('table', 'data'),
              [dash.dependencies.Input('submit-val', 'n_clicks'),
               dash.dependencies.Input('table', 'data_previous'),
               dash.dependencies.Input('table', 'data')],
              [dash.dependencies.State('date-range', 'start_date')],
              [dash.dependencies.State('date-range', 'end_date')]
              )
def update_table(n_clicks, predictions, data, start_date, end_date):
    predictions = pd.DataFrame.from_dict(predictions)
    data = pd.DataFrame.from_dict(data)
    data = data[(data.Date >= start_date) &
                (data.Date < end_date)]

    predictions = predictions[(predictions.Date >= start_date) &
                              (predictions.Date < end_date)]
    return data.append(predictions).to_dict('records')

@app.callback(dash.dependencies.Output('predictions-graphs', 'figure'),
              dash.dependencies.Input('table', 'data')
              )
def update_predictions_graphs(data):
    predictions = pd.DataFrame.from_records(data)

    fig = px.line(predictions,
        facet_col="Prescriptor",
        color="Prescriptor",
        line_group="PrescriptionIndex",
        x="Date",
        y="PredictedDailyNewCases",
        facet_col_wrap=3)

    return fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8050, host='0.0.0.0')
