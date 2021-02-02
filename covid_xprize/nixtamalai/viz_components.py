import pandas as pd
import plotly.express as px
from covid_xprize.scoring.prescriptor_scoring import compute_pareto_set
from covid_xprize.nixtamalai.prescriptors import get_greedy_prescription_df
from covid_xprize.nixtamalai.prescriptors import generate_cases_and_stringency_for_prescriptions



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
    costs = {k: round(float(vals[k])/IP_MAX_VALUES[k],1) for k in vals.keys()}
    return costs

def npi_cost_to_val(costs:dict):
    vals = {k: ([round(costs[k][0]*IP_MAX_VALUES[k])]
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

def get_overall_data(start_date, end_date, ip_file, weights_df):
    prescription_df = get_greedy_prescription_df(start_date, end_date, ip_file,weights_df)
    df, _ = generate_cases_and_stringency_for_prescriptions(start_date,
                                                            end_date,
                                                            prescription_df,
                                                            weights_df)
    overall_pdf = df.groupby('PrescriptionIndex').mean().reset_index()
    return overall_pdf
