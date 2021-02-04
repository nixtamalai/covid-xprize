import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dash
import dash_core_components as dcc
import dash_extendable_graph as deg
import plotly.graph_objects as go
import dash_html_components as html
import plotly.express as px
import plotly.io as pio
from covid_xprize.nixtamalai.viz_components import get_pareto_data
from covid_xprize.nixtamalai.viz_components import npi_val_to_cost
from covid_xprize.nixtamalai.viz_components import get_overall_data
from covid_xprize.nixtamalai.viz_components import npi_cost_to_val
import palettable as pltt


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
TEMPLATE = "ggplot2"


START_DATE = "2020-08-01"
END_DATE = "2020-09-01"
TEST_COST = "covid_xprize/validation/data/uniform_random_costs.csv"
COUNTRY = "Mexico"
# Este escenario sólo trae a México, por eso sólo se modela ese geo. Esto tendría que cambiar
IP_FILE = "prescriptions/robojudge_test_scenario.csv"
DEFAULT_COLORS = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]

weights_df = pd.read_csv(TEST_COST, keep_default_na=False)
# Filtro por el país "seleccionado"
weights_df = weights_df[weights_df.CountryName == "Mexico"]
overall_pdf, predictions = get_overall_data(START_DATE, END_DATE, IP_FILE, weights_df)
# Gráfica inicial de Pareto
pareto = get_pareto_data(list(overall_pdf['Stringency']),
                         list(overall_pdf['PredictedDailyNewCases']))
pareto_data = {"x": pareto[0],
               "y": pareto[1],
               "name": "Base Prescriptor",
               "showlegend": True,
               }
# valores de pesos para popular los sliders
# npis = (weights_df
#         .drop(columns=['CountryName', 'RegionName'])
#         .to_dict(orient='records'))[0]
npis = (weights_df
        .drop(columns=['CountryName', 'RegionName', 'GeoID'])
        .to_dict(orient='records'))[0]
BASE_COSTS = npi_val_to_cost(npis)
# Gráfica inicial de radar
radar_data = {
    "r": [v for _,v in BASE_COSTS.items()],
    "theta": [k.split("_")[0] for k,_ in BASE_COSTS.items()],
    "name": "Base Prescriptor",
    'type': 'scatterpolar',
    "showlegend": True,
}
# Gráfica inicial  de predicciones
predictions = pd.concat(predictions)
fig_predictions = go.Figure(layout={"title": {"text": "Predictions plot"},
                            "xaxis": {"title": "Date"},
                            "yaxis": {"title": "New Cases per Day"},
                            "template": TEMPLATE
                            })
for idx in predictions.PrescriptionIndex.unique():
    display_legend = True if idx == 0 else False
    idf = predictions[predictions.PrescriptionIndex == idx]
    fig_predictions.add_trace(
        go.Scatter(
            x=idf["Date"],
            y=idf["PredictedDailyNewCases"],
            mode='lines', line=dict(color=DEFAULT_COLORS[1]),
            name="Base prescription",
            legendgroup="group_0",
            showlegend=display_legend
        )
    )

fig_predictions_heat = go.Figure(
    layout={"title": {"text": "Predictions plot"},
            "xaxis": {"title": "Date"},
            "yaxis": {"title": "New Cases per Day"},
            "template": TEMPLATE})

for idx in predictions.PrescriptionIndex.unique():
    display_legend = True if idx == 0 else False
    idf = predictions[predictions.PrescriptionIndex == idx]
    fig_predictions_heat.add_trace(
        go.Scatter(
            x=idf["Date"],
            y=idf["PredictedDailyNewCases"],
            mode='lines', line=dict(color=DEFAULT_COLORS[1]),
            name="Base prescription",
            legendgroup="group_0",
            showlegend=display_legend
        )
    )

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='Visualizing Intervention Plans'),

    # html.Div(children='''
    #     XPRIZE
    # '''),
    html.Div(
        children =[
            deg.ExtendableGraph(
                id='pareto-plot',
                figure=go.Figure(dict(
                    data=[pareto_data],
                    layout={"title": {"text": "Pareto plot"},
                            "xaxis": {"title": "Mean Stringency"},
                            "yaxis": {"title": "Mean New Cases per Day"},
                            "legend":{"yanchor": "top","y": 0.99, "x": 0.8},
                            "template": TEMPLATE
                            }
                ))
            )
    ],
    style={'width': '50%', 'height':'50%', 'display': 'inline-block',"float":"left" }
    ),
    html.Div(
        children =[
            deg.ExtendableGraph(
                id='predictions-plot',
                figure=fig_predictions
            )
    ],
    style={'width': '50%', 'height':'50%', "float":"left"}
    ),

    html.Div(
        children =[
            html.P(children=dcc.Markdown("School closing (**C1**)")),
            dcc.Slider(
                id='C1-weight',
                min=0,
                max=1,
                step=0.1,
                value=BASE_COSTS['C1_School closing'],
                marks = {0:"0", 0.5:"0.5", 1: "1"},
                tooltip = { 'always_visible': False }
            ),
            html.P(children=dcc.Markdown("Workplace closing (**C2**)")),
            dcc.Slider(
                id='C2-weight',
                min=0,
                max=1,
                step=0.1,
                value=BASE_COSTS['C2_Workplace closing'],
                marks = {0:"0", 0.5:"0.5", 1: "1"},
                tooltip = { 'always_visible': False }

            ),
            html.P(children=dcc.Markdown("Cancel public events (**C3**)")),
            dcc.Slider(
                id='C3-weight',
                min=0,
                max=1,
                step=0.1,
                value=BASE_COSTS['C3_Cancel public events'],
                marks = {0:"0", 0.5:"0.5", 1: "1"},
                tooltip = { 'always_visible': False }

            ),
            html.P(children=dcc.Markdown("Restrictions on gathering (**C4**)")),
            dcc.Slider(
                id='C4-weight',
                min=0,
                max=1,
                step=0.1,
                value=BASE_COSTS['C4_Restrictions on gatherings'],
                marks = {0:"0", 0.5:"0.5", 1: "1"},
                tooltip = { 'always_visible': False }

            ),
            html.P(children=dcc.Markdown("Close public transport (**C5**)")),
            dcc.Slider(
                id='C5-weight',
                min=0,
                max=1,
                step=0.1,
                value=BASE_COSTS['C5_Close public transport'],
                marks = {0:"0", 0.5:"0.5", 1: "1"},
                tooltip = { 'always_visible': False }

            ),
            html.P(children=dcc.Markdown("Stay at home requirements (**C6**)")),
            dcc.Slider(
                id='C6-weight',
                min=0,
                max=1,
                step=0.1,
                value=BASE_COSTS['C6_Stay at home requirements'],
                marks = {0:"0", 0.5:"0.5", 1: "1"},
                tooltip = { 'always_visible': False }

            ),
            html.Button('Reset', id='reset-val', n_clicks=0)
            ],
            style={'width': '20%', 'height':'30%','display': 'inline-block',
                   "background-color": "rgb(237, 237, 237)", "margin-left":"60px"}),
    html.Div(
             children=[
                deg.ExtendableGraph(
                 id='predictions-heat-plot',
                 figure=fig_predictions)],
             style={'width': '50%', 'height': '50%', "float": "left"}
    ),
    html.Div(
        children=[
            html.P(children=dcc.Markdown("Restrictions on internal movement (**C7**)")),
            dcc.Slider(
                id='C7-weight',
                min=0,
                max=1,
                step=0.1,
                value=BASE_COSTS['C7_Restrictions on internal movement'],
                marks = {0:"0", 0.5:"0.5", 1: "1"},
                tooltip = { 'always_visible': False }

            ),
            html.P(children=dcc.Markdown("International travel conrols (**C8**)")),
            dcc.Slider(
                id='C8-weight',
                min=0,
                max=1,
                step=0.1,
                value=BASE_COSTS['C8_International travel controls'],
                marks = {0:"0", 0.5:"0.5", 1: "1"},
                tooltip = { 'always_visible': False }

            ),
            html.P(children=dcc.Markdown("Public information campaigns (**H1**)")),
            dcc.Slider(
                id='H1-weight',
                min=0,
                max=1,
                step=0.1,
                value=BASE_COSTS['H1_Public information campaigns'],
                marks = {0:"0", 0.5:"0.5", 1: "1"},
                tooltip = { 'always_visible': False }

            ),
            html.P(children=dcc.Markdown("Testing policy (**H2**)")),
            dcc.Slider(
                id='H2-weight',
                min=0,
                max=1,
                step=0.1,
                value=BASE_COSTS['H2_Testing policy'],
                marks = {0:"0", 0.5:"0.5", 1: "1"},
                tooltip = { 'always_visible': False }

            ),
            html.P(children=dcc.Markdown("Contact tracing (**H3**)")),
            dcc.Slider(
                id='H3-weight',
                min=0,
                max=1,
                step=0.1,
                value=BASE_COSTS['H3_Contact tracing'],
                marks = {0:"0", 0.5:"0.5", 1: "1"},
                tooltip = { 'always_visible': False }

            ),
            html.P(children=dcc.Markdown("Facial coverings (**H4**)")),
            dcc.Slider(
                id='H4-weight',
                min=0,
                max=1,
                step=0.1,
                value=BASE_COSTS['H6_Facial Coverings'],
                marks = {0:"0", 0.5:"0.5", 1: "1"},
                tooltip = { 'always_visible': False }

            ),
            html.Button('Submit', id='submit-val', n_clicks=0)
    ],
    style={'width': '20%', 'height':'30%','display': 'inline-block', 'margin-top':0,
           "background-color": "rgb(237, 237, 237)"}
    ),
    # html.Div(
    #     children=[html.Button('Submit', id='submit-val', n_clicks=0)],
    #     style={'display': 'inline-block', "float":"right"}
    # ),
    html.Div(
        children=[deg.ExtendableGraph(
            id='radar-plot',
            figure=go.Figure(dict(
                data=[radar_data],
                layout={"title": {"text": "NPI Weights"},
                        "template": TEMPLATE}
            ))
        )],
        style={'width': '35%', 'display': 'inline-block', "float": "right"}
    ),
])


@app.callback([dash.dependencies.Output('pareto-plot', 'extendData'),
               dash.dependencies.Output('predictions-plot', 'extendData'),
               dash.dependencies.Output('predictions-heat-plot', 'extendData')],
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
               [dash.dependencies.State('pareto-plot', 'figure')]
              )
def update_pareto_plot(n_clicks, value_c1, value_c2, value_c3, value_c4, value_c5, value_c6,
               value_c7, value_c8, value_h1, value_h2, value_h3, value_h4, figure):
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
        weights_dict = npi_cost_to_val(weights_dict)
        user_weights = pd.DataFrame.from_dict(weights_dict)
        overall_pdf, predictions = get_overall_data(
            START_DATE, END_DATE, IP_FILE, user_weights)
        pareto = get_pareto_data(list(overall_pdf['Stringency']),
                                 list(overall_pdf['PredictedDailyNewCases']))
        new_trace = {"x": pareto[0],
                     "y": pareto[1],
                     "name": "User prescription {}".format(n_clicks)}
        predictions = pd.concat(predictions)
        prediction_traces = []
        for idx in predictions.PrescriptionIndex.unique():
            display_legend = True if idx == 0 else False
            idf = predictions[predictions.PrescriptionIndex == idx]
            trace = {"x": idf["Date"],
                     "y": idf["PredictedDailyNewCases"],
                     "mode": "lines",
                     "line": dict(color=DEFAULT_COLORS[n_clicks + 1]),
                     "name": "User prescription {}".format(n_clicks),
                     "legendgroup": "group_{}".format(n_clicks),
                     "showlegend": display_legend
                    }
            prediction_traces.append(trace)
            # fig_predictions.add_trace(go.Scatter(x=idf["Date"], y=idf["PredictedDailyNewCases"],
            #                 mode='lines',line=dict(color=DEFAULT_COLORS[1])))

        return ([new_trace, []], []), (([prediction_traces, []], [])), (([prediction_traces, []], []))
    return ([],[],[]), ([],[],[]), ([],[],[])

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
               [dash.dependencies.State('radar-plot', 'figure')]
              )
def update_radar_plot(n_clicks, value_c1, value_c2, value_c3, value_c4, value_c5, value_c6,
               value_c7, value_c8, value_h1, value_h2, value_h3, value_h4, figure):
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
        new_trace = {
            "r": [v for _,v in weights_dict.items()],
            "theta": [k for k,_ in weights_dict.items()],
            'type': 'scatterpolar',
            "name": "User prescription {}".format(n_clicks)
        }
        return [new_trace, []], []

if __name__ == '__main__':
    app.run_server(debug=True, port=8051, host='0.0.0.0')
