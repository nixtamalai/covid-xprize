import pandas as pd
import dash_html_components as html
import dash_core_components as dcc
from covid_xprize.scoring.prescriptor_scoring import compute_pareto_set
from covid_xprize.nixtamalai.prescriptors import get_greedy_prescription_df
from covid_xprize.nixtamalai.prescriptors import generate_cases_and_stringency_for_prescriptions
from covid_xprize.nixtamalai import surrogate_model

# No entiendo muy bien estos max values, la verdad
IP_MAX_VALUES = {
    'C1_School closing': 3,
    'C2_Workplace closing': 3,
    'C3_Cancel public events': 2,
    'C4_Restrictions on gatherings': 4,
    'C5_Close public transport': 2,
    'C6_Stay at home requirements': 3,
    'C7_Restrictions on internal movement': 2,
    'C8_International travel controls': 4,
    'H1_Public information campaigns': 2,
    'H2_Testing policy': 3,
    'H3_Contact tracing': 2,
    'H6_Facial Coverings': 4
}

def npi_val_to_cost(vals:dict):
    costs = {k: float(vals[k])/IP_MAX_VALUES[k] for k in vals.keys()}
    return costs

def npi_cost_to_val(costs:dict):
    vals = {k: ([costs[k][0]*IP_MAX_VALUES[k]]
                if k not in ['CountryName', 'RegionName'] else costs[k]) for k in costs.keys()}
    return vals


def get_pareto_data(objective1_list, objective2_list):
    """
    Plot the pareto curve given the objective values for a set of solutions.
    This curve indicates the area dominated by the solution set, i.e., 
    every point up and to the right is dominated.
    """
    
    # Compute pareto set from full solution set.
    objective1_pareto, objective2_pareto = compute_pareto_set(objective1_list, 
                                                              objective2_list)
    
    # Sort by first objective.
    objective1_pareto, objective2_pareto = list(zip(*sorted(zip(objective1_pareto,
                                                                objective2_pareto))))
    
    # Compute the coordinates to plot.
    xs = []
    ys = []
    
    xs.append(objective1_pareto[0])
    ys.append(objective2_pareto[0])
    
    for i in range(0, len(objective1_pareto)-1):
        
        # Add intermediate point between successive solutions
        xs.append(objective1_pareto[i+1])
        ys.append(objective2_pareto[i])
        
        # Add next solution on front
        xs.append(objective1_pareto[i+1])
        ys.append(objective2_pareto[i+1])
        
    # df = pd.DataFrame([xs, ys]).T
    # df.columns = ["Stringency", 'PredictedDailyNewCases']
    # return px.line(df, x="Stringency", y='PredictedDailyNewCases', color_discrete_sequence=[color])
    return xs, ys

def get_overall_data(start_date, end_date, ip_file, weights_df, model):
    if model == "nixtamal":
        prescription_df = surrogate_model.prescribe(start_date, end_date, ip_file, weights_df)
    else:
        prescription_df = get_greedy_prescription_df(start_date, end_date, ip_file, weights_df)
    df, predictions = generate_cases_and_stringency_for_prescriptions(start_date,
                                                            end_date,
                                                            prescription_df,
                                                            weights_df)
    overall_pdf = df.groupby('PrescriptionIndex').mean().reset_index()
    return overall_pdf, predictions

def get_sliders(BASE_COSTS):
    sliders = [
        html.Div([html.P(children=dcc.Markdown("School closing (**C1**)")),
        dcc.Slider(
            id='C1-weight',
            min=0,
            max=1,
            step=0.1,
            value=BASE_COSTS['C1_School closing'],
            marks={0: "0", 0.5: "0.5", 1: "1"},
            tooltip={'always_visible': False}
        )]),
        html.Div([html.P(children=dcc.Markdown("Workplace closing (**C2**)")),
        dcc.Slider(
            id='C2-weight',
            min=0,
            max=1,
            step=0.1,
            value=BASE_COSTS['C2_Workplace closing'],
            marks={0: "0", 0.5: "0.5", 1: "1"},
            tooltip={'always_visible': False}

        )]),
        html.Div([html.P(children=dcc.Markdown("Cancel public events (**C3**)")),
        dcc.Slider(
            id='C3-weight',
            min=0,
            max=1,
            step=0.1,
            value=BASE_COSTS['C3_Cancel public events'],
            marks={0: "0", 0.5: "0.5", 1: "1"},
            tooltip={'always_visible': False}

        )]),
        html.Div([html.P(children=dcc.Markdown("Restrictions on gathering (**C4**)")),
        dcc.Slider(
            id='C4-weight',
            min=0,
            max=1,
            step=0.1,
            value=BASE_COSTS['C4_Restrictions on gatherings'],
            marks={0: "0", 0.5: "0.5", 1: "1"},
            tooltip={'always_visible': False}

        )]),
        html.Div([html.P(children=dcc.Markdown("Close public transport (**C5**)")),
        dcc.Slider(
            id='C5-weight',
            min=0,
            max=1,
            step=0.1,
            value=BASE_COSTS['C5_Close public transport'],
            marks={0: "0", 0.5: "0.5", 1: "1"},
            tooltip={'always_visible': False}

        )]),
        html.Div([html.P(children=dcc.Markdown("Stay at home requirements (**C6**)")),
        dcc.Slider(
            id='C6-weight',
            min=0,
            max=1,
            step=0.1,
            value=BASE_COSTS['C6_Stay at home requirements'],
            marks={0: "0", 0.5: "0.5", 1: "1"},
            tooltip={'always_visible': False}

        )]),
        html.Div([html.P(children=dcc.Markdown(
            "Restrictions on internal movement (**C7**)")),
        dcc.Slider(
            id='C7-weight',
            min=0,
            max=1,
            step=0.1,
            value=BASE_COSTS['C7_Restrictions on internal movement'],
            marks={0: "0", 0.5: "0.5", 1: "1"},
            tooltip={'always_visible': False}

        )]),
        html.Div([html.P(children=dcc.Markdown("International travel controls (**C8**)")),
        dcc.Slider(
            id='C8-weight',
            min=0,
            max=1,
            step=0.1,
            value=BASE_COSTS['C8_International travel controls'],
            marks={0: "0", 0.5: "0.5", 1: "1"},
            tooltip={'always_visible': False}

        )]),
        html.Div([html.P(children=dcc.Markdown("Public information campaigns (**H1**)")),
        dcc.Slider(
            id='H1-weight',
            min=0,
            max=1,
            step=0.1,
            value=BASE_COSTS['H1_Public information campaigns'],
            marks={0: "0", 0.5: "0.5", 1: "1"},
            tooltip={'always_visible': False}

        )]),
        html.Div([html.P(children=dcc.Markdown("Testing policy (**H2**)")),
        dcc.Slider(
            id='H2-weight',
            min=0,
            max=1,
            step=0.1,
            value=BASE_COSTS['H2_Testing policy'],
            marks={0: "0", 0.5: "0.5", 1: "1"},
            tooltip={'always_visible': False}

        )]),
        html.Div([html.P(children=dcc.Markdown("Contact tracing (**H3**)")),
        dcc.Slider(
            id='H3-weight',
            min=0,
            max=1,
            step=0.1,
            value=BASE_COSTS['H3_Contact tracing'],
            marks={0: "0", 0.5: "0.5", 1: "1"},
            tooltip={'always_visible': False}

        )]),
        html.Div([html.P(children=dcc.Markdown("Facial coverings (**H4**)")),
        dcc.Slider(
            id='H4-weight',
            min=0,
            max=1,
            step=0.1,
            value=BASE_COSTS['H6_Facial Coverings'],
            marks={0: "0", 0.5: "0.5", 1: "1"},
            tooltip={'always_visible': False}
        )])
        ]
    return sliders


def get_sliders_2(BASE_COSTS):
    sliders = [
        html.P(children=dcc.Markdown(
            "Restrictions on internal movement (**C7**)")),
        dcc.Slider(
            id='C7-weight',
            min=0,
            max=1,
            step=0.1,
            value=BASE_COSTS['C7_Restrictions on internal movement'],
            marks={0: "0", 0.5: "0.5", 1: "1"},
            tooltip={'always_visible': False}

        ),
        html.P(children=dcc.Markdown("International travel controls (**C8**)")),
        dcc.Slider(
            id='C8-weight',
            min=0,
            max=1,
            step=0.1,
            value=BASE_COSTS['C8_International travel controls'],
            marks={0: "0", 0.5: "0.5", 1: "1"},
            tooltip={'always_visible': False}

        ),
        html.P(children=dcc.Markdown("Public information campaigns (**H1**)")),
        dcc.Slider(
            id='H1-weight',
            min=0,
            max=1,
            step=0.1,
            value=BASE_COSTS['H1_Public information campaigns'],
            marks={0: "0", 0.5: "0.5", 1: "1"},
            tooltip={'always_visible': False}

        ),
        html.P(children=dcc.Markdown("Testing policy (**H2**)")),
        dcc.Slider(
            id='H2-weight',
            min=0,
            max=1,
            step=0.1,
            value=BASE_COSTS['H2_Testing policy'],
            marks={0: "0", 0.5: "0.5", 1: "1"},
            tooltip={'always_visible': False}

        ),
        html.P(children=dcc.Markdown("Contact tracing (**H3**)")),
        dcc.Slider(
            id='H3-weight',
            min=0,
            max=1,
            step=0.1,
            value=BASE_COSTS['H3_Contact tracing'],
            marks={0: "0", 0.5: "0.5", 1: "1"},
            tooltip={'always_visible': False}

        ),
        html.P(children=dcc.Markdown("Facial coverings (**H4**)")),
        dcc.Slider(
            id='H4-weight',
            min=0,
            max=1,
            step=0.1,
            value=BASE_COSTS['H6_Facial Coverings'],
            marks={0: "0", 0.5: "0.5", 1: "1"},
            tooltip={'always_visible': False}
        )]
    return sliders
