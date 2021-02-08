import os
from datetime import date
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dash
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

TEMPLATE = 'plotly_dark'


START_DATE = "2020-08-01"
END_DATE = "2020-09-01"
TEST_COST = "covid_xprize/validation/data/uniform_random_costs.csv"
COUNTRY = "Mexico"
# Este escenario sólo trae a México, por eso sólo se modela ese geo. Esto tendría que cambiar
IP_FILE = "prescriptions/robojudge_test_scenario.csv"
DEFAULT_COLORS = px.colors.qualitative.Plotly
weights_df = pd.read_csv(TEST_COST, keep_default_na=False)
# Filtro por el país "seleccionado"
weights_df = weights_df[weights_df.CountryName == "Mexico"]
overall_pdf, predictions = get_overall_data(START_DATE, END_DATE, IP_FILE, weights_df, "greedy")
# Gráfica inicial de Pareto
pareto = get_pareto_data(list(overall_pdf['Stringency']),
                         list(overall_pdf['PredictedDailyNewCases']))
pareto_data = {"x": pareto[0],
               "y": pareto[1],
               "name": "Base (Blind Greedy)",
               "showlegend": True,
               }
npis = (weights_df
        .drop(columns=['CountryName', 'RegionName', 'GeoID'])
        .to_dict(orient='records'))[0]
BASE_COSTS = npi_val_to_cost(npis)
# Gráfica inicial de radar
radar_data = {
    "r": [v for _,v in BASE_COSTS.items()],
    "theta": [k.split("_")[0] for k,_ in BASE_COSTS.items()],
    "name": "Base (Blind Greedy)",
    'type': 'scatterpolar',
    "showlegend": True,
}
# Gráfica inicial  de predicciones
predictions = pd.concat(predictions)
fig_predictions = go.Figure(layout={ 
                            "xaxis": {"title": "Date"},
                            "yaxis": {"title": "New Cases per Day"},
                            "legend": {"yanchor": "top", "y": 0.99, "x": 0.05},
                            "template": TEMPLATE
                            })
for idx in predictions.PrescriptionIndex.unique():
    display_legend = True if idx == 0 else False
    idf = predictions[predictions.PrescriptionIndex == idx]
    fig_predictions.add_trace(
        go.Scatter(
            x=idf["Date"],
            y=idf["PredictedDailyNewCases"],
            mode='lines', line=dict(color=DEFAULT_COLORS[0]),
            name="Base (Blind Greedy)",
            legendgroup="group_0",
            showlegend=display_legend
        )
    )

sliders = get_sliders(BASE_COSTS)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

app.layout =html.Div(
    [
        dbc.Row(dbc.Col(html.Div(html.H1(children='Visualizing Intervention Plans')))),
        dbc.Row(html.Hr()),
        dbc.Row(
            [
                dbc.Col(html.Div(sliders[0:3]), width={"size": 2, "offset": 1}),
                dbc.Col(html.Div(sliders[3:6]), width=2),
                dbc.Col(html.Div(sliders[6:9]), width=2),
                dbc.Col(html.Div(sliders[9:12]), width=2),
                dbc.Col(
                    [
                        html.Hr(),
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
                        ))                       
                    ],
                        
                    width=2),
                dbc.Col(
                    [
                        html.Hr(),
                        html.Div(dbc.Button('Submit', id='submit-val',color="success",
                          n_clicks=0, block=True)),
                        html.Hr(),
                        html.Div(dbc.Button('Reset', id='reset-val', color="warning", 
                        n_clicks=0, block=True))
                        
                    ],
                        
                    width=1),
            ]
        ),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    html.Div(html.H4("NPI Weights"))
                    ),               
                dbc.Col(
                    html.Div(html.H4("Pareto Plot"))
                    ),
                dbc.Col(
                    html.Div(html.H4("Predictions"))
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
                                layout={"title": {"text": "NPI Weights"},
                                        "template": TEMPLATE}
                            ))
                    ), width=4
                ),
                dbc.Col(
                    deg.ExtendableGraph(
                        id='pareto-plot',
                        figure=go.Figure(dict(
                            data=[pareto_data],
                            layout={
                                    "xaxis": {"title": "Mean Stringency"},
                                    "yaxis": {"title": "Mean New Cases per Day"},
                                    "legend": {"yanchor": "top", "y": 0.99, "x": 0.5},
                                    "template": TEMPLATE
                                    }
                        ))
                    ), width=4
                ),
                dbc.Col(
                    deg.ExtendableGraph(
                        id='predictions-plot',
                        figure=fig_predictions
                    ), width=4
                )
            ],
            align="center",
        ),
    ]
)

@app.callback([dash.dependencies.Output('pareto-plot', 'extendData'),
               dash.dependencies.Output('predictions-plot', 'extendData')],
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
               [dash.dependencies.State('date-range', 'start_date')],
               [dash.dependencies.State('date-range', 'end_date')],
               [dash.dependencies.State('pareto-plot', 'figure')]
              )
def update_pareto_plot(n_clicks, value_c1, value_c2, value_c3, value_c4, value_c5, value_c6,
               value_c7, value_c8, value_h1, value_h2, value_h3, value_h4, model, start_date, 
               end_date, figure):
    if n_clicks > 0:    
        weights_dict = {
            'CountryName': ['Mexico'],
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
            start_date, end_date, IP_FILE, user_weights, model)
        pareto = get_pareto_data(list(overall_pdf['Stringency']),
                                 list(overall_pdf['PredictedDailyNewCases']))
        new_trace = {"x": pareto[0],
                     "y": pareto[1],
                     "name": "{} prescription {}".format(prescriptor_names[model], n_clicks)}
        predictions = pd.concat(predictions)
        prediction_traces = []
        for idx in predictions.PrescriptionIndex.unique():
            display_legend = True if idx == 0 else False
            idf = predictions[predictions.PrescriptionIndex == idx]
            trace = {"x": idf["Date"],
                     "y": idf["PredictedDailyNewCases"],
                     "mode": "lines",
                     "line": dict(color=DEFAULT_COLORS[n_clicks]),
                     "name": "{} prescription {}".format(prescriptor_names[model], n_clicks),
                     "legendgroup": "group_{}".format(n_clicks),
                     "showlegend": display_legend
                    }
            prediction_traces.append(trace) 
        return ([new_trace, []], []), (([prediction_traces, []], []))
    return ([],[],[]), ([],[],[])

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
               [dash.dependencies.State('radar-plot', 'figure')]
              )
def update_radar_plot(n_clicks, value_c1, value_c2, value_c3, value_c4, value_c5, value_c6,
               value_c7, value_c8, value_h1, value_h2, value_h3, value_h4, model, figure):
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
            "name": "{} prescription {}".format(prescriptor_names[model], n_clicks)
        }
        return [new_trace, []], []


if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
