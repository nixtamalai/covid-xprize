import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dash
import dash_core_components as dcc
import dash_extendable_graph as deg
import dash_html_components as html
import plotly.express as px
from covid_xprize.nixtamalai.viz_components import get_pareto_data
from covid_xprize.nixtamalai.viz_components import npi_val_to_cost
from covid_xprize.nixtamalai.viz_components import get_overall_data
from covid_xprize.nixtamalai.viz_components import npi_cost_to_val
import palettable as pltt



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']



START_DATE = "2020-08-01"
END_DATE = "2020-08-05"
TEST_COST = "covid_xprize/validation/data/uniform_random_costs.csv"
IP_FILE = "prescriptions/robojudge_test_scenario.csv"

weights_df = pd.read_csv(TEST_COST, keep_default_na=False)
overall_pdf = get_overall_data(START_DATE, END_DATE, IP_FILE, weights_df)
pareto = get_pareto_data(list(overall_pdf['Stringency']),
                         list(overall_pdf['PredictedDailyNewCases']))
pareto_data = {"x": pareto[0],
               "y": pareto[1],
               "name": "Base Prescriptor"
               }
# valores de pesos para popular los sliders
npis = (pd.read_csv(TEST_COST)
        .drop(columns=['CountryName', 'RegionName'])
        .to_dict(orient='records'))[0]
costs = npi_val_to_cost(npis)

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='Visualizing Intervention Plans'),

    html.Div(children='''
        XPRIZE
    '''),
    html.Div(
        children =[
            deg.ExtendableGraph(
                id='pareto-plot',
                figure=dict(
                    data=[pareto_data],
                    layout={"title": {"text": "Pareto plot"}, 
                            "xaxis": {"title": "Mean Stringency"},
                            "yaxis": {"title": "Mean New Cases per Day"} }
                )
            )
    ],
    style={'width': '60%', 'height':'50%', 'display': 'inline-block'}
    ),
    html.Div(
        children =[
            html.P(children="School closing"),
            dcc.Slider(
                id='C1-weight',
                min=0,
                max=1,
                step=0.1,
                value=costs['C1_School closing'],
                marks = {0:"0", 0.5:"0.5", 1: "1"},
                tooltip = { 'always_visible': False }
            ),
            html.P(children="Workplace closing"),
            dcc.Slider(
                id='C2-weight',
                min=0,
                max=1,
                step=0.1,
                value=costs['C2_Workplace closing'],
                marks = {0:"0", 0.5:"0.5", 1: "1"},
                tooltip = { 'always_visible': False }

            ),
            html.P(children="Cancel public events"),
            dcc.Slider(
                id='C3-weight',
                min=0,
                max=1,
                step=0.1,
                value=costs['C3_Cancel public events'],
                marks = {0:"0", 0.5:"0.5", 1: "1"},
                tooltip = { 'always_visible': False }

            ),
            html.P(children="Restrictions on gathering"),
            dcc.Slider(
                id='C4-weight',
                min=0,
                max=1,
                step=0.1,
                value=costs['C4_Restrictions on gatherings'],
                marks = {0:"0", 0.5:"0.5", 1: "1"},
                tooltip = { 'always_visible': False }

            ),
            html.P(children="Close public transport"),
            dcc.Slider(
                id='C5-weight',
                min=0,
                max=1,
                step=0.1,
                value=costs['C5_Close public transport'],
                marks = {0:"0", 0.5:"0.5", 1: "1"},
                tooltip = { 'always_visible': False }

            ),
            html.P(children="Stay at home requirements"),
            dcc.Slider(
                id='C6-weight',
                min=0,
                max=1,
                step=0.1,
                value=costs['C6_Stay at home requirements'],
                marks = {0:"0", 0.5:"0.5", 1: "1"},
                tooltip = { 'always_visible': False }

            )],
            style={'width': '20%', 'display': 'inline-block'}),
    html.Div(
        children=[
            html.P(children="Restrictions on internal movement"),
            dcc.Slider(
                id='C7-weight',
                min=0,
                max=1,
                step=0.1,
                value=costs['C7_Restrictions on internal movement'],
                marks = {0:"0", 0.5:"0.5", 1: "1"},
                tooltip = { 'always_visible': False }

            ),
            html.P(children="International travel conrols"),
            dcc.Slider(
                id='C8-weight',
                min=0,
                max=1,
                step=0.1,
                value=costs['C8_International travel controls'],
                marks = {0:"0", 0.5:"0.5", 1: "1"},
                tooltip = { 'always_visible': False }

            ),
            html.P(children="Public information campaigns"),
            dcc.Slider(
                id='H1-weight',
                min=0,
                max=1,
                step=0.1,
                value=costs['H1_Public information campaigns'],
                marks = {0:"0", 0.5:"0.5", 1: "1"},
                tooltip = { 'always_visible': False }

            ),
            html.P(children="Testing policy"),
            dcc.Slider(
                id='H2-weight',
                min=0,
                max=1,
                step=0.1,
                value=costs['H2_Testing policy'],
                marks = {0:"0", 0.5:"0.5", 1: "1"},
                tooltip = { 'always_visible': False }

            ),
            html.P(children="Contact tracing"),
            dcc.Slider(
                id='H3-weight',
                min=0,
                max=1,
                step=0.1,
                value=costs['H3_Contact tracing'],
                marks = {0:"0", 0.5:"0.5", 1: "1"},
                tooltip = { 'always_visible': False }

            ),
            html.P(children="Facial coverings"),
            dcc.Slider(
                id='H4-weight',
                min=0,
                max=1,
                step=0.1,
                value=costs['H6_Facial Coverings'],
                marks = {0:"0", 0.5:"0.5", 1: "1"},
                tooltip = { 'always_visible': False }

            )
    ],
    style={'width': '20%', 'display': 'inline-block', 'margin-top':0}
    ),
    html.Div(
        children=[html.Button('Submit', id='submit-val', n_clicks=0)],
        style={'width': '20%', 'display': 'inline-block', "float":"right"}
    ),
    # html.Div(id='container-button-basic',
    #          children='Enter a value and press submit')
])


@app.callback(dash.dependencies.Output('pareto-plot', 'extendData'),
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
        overall_pdf = get_overall_data(
            START_DATE, END_DATE, IP_FILE, user_weights)
        pareto = get_pareto_data(list(overall_pdf['Stringency']),
                                 list(overall_pdf['PredictedDailyNewCases']))
        new_trace = {"x": pareto[0],
                     "y": pareto[1],
                     "name": "User prescription {}".format(n_clicks)}        
        return [new_trace, []], []



if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
