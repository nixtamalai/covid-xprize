import os
import argparse
import numpy as np
import pandas as pd
import time
from covid_xprize.standard_predictor.xprize_predictor import XPrizePredictor
from covid_xprize.standard_predictor.xprize_predictor import NPI_COLUMNS
from covid_xprize.nixtamalai.helpers import add_geo_id


NUM_PRESCRIPTIONS = 10

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

IP_COLS = list(IP_MAX_VALUES.keys())


def get_greedy_prescription_df(start_date_str: str,
              end_date_str: str,
              hist_df: pd.DataFrame,
              weights_df: pd.DataFrame) -> pd.DataFrame:

    # Load historical IPs, just to extract the geos
    # we need to prescribe for.
    # hist_df = pd.read_csv(path_to_hist_file,
    #                       parse_dates=['Date'],
    #                       encoding="ISO-8859-1",
    #                       keep_default_na=False,
    #                       error_bad_lines=True)

    # Generate prescriptions
    start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
    end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')
    prescription_dict = {
        'PrescriptionIndex': [],
        'CountryName': [],
        'RegionName': [],
        'Date': []
    }
    for ip in IP_COLS:
        prescription_dict[ip] = []
    for country_name in hist_df['CountryName'].unique():
        country_df = hist_df[hist_df['CountryName'] == country_name]
        #print(country_df)
        for region_name in country_df['RegionName'].unique():
            # Sort IPs for this geo by weight
            if isinstance(region_name, float):
                geo_weights_df = weights_df[(weights_df['CountryName'] == country_name)][IP_COLS]
            else:    
                geo_weights_df = weights_df[(weights_df['CountryName'] == country_name) &
                                            (weights_df['RegionName'] == region_name)][IP_COLS]
            ip_names = list(geo_weights_df.columns)
            ip_weights = geo_weights_df.values[0]
            sorted_ips = [ip for _, ip in sorted(zip(ip_weights, ip_names))]

            # Initialize the IPs to all turned off
            curr_ips = {ip: 0 for ip in IP_MAX_VALUES}

            for prescription_idx in range(NUM_PRESCRIPTIONS):

                # Turn on the next IP
                next_ip = sorted_ips[prescription_idx]
                curr_ips[next_ip] = IP_MAX_VALUES[next_ip]

                # Use curr_ips for all dates for this prescription
                for date in pd.date_range(start_date, end_date):
                    prescription_dict['PrescriptionIndex'].append(prescription_idx)
                    prescription_dict['CountryName'].append(country_name)
                    prescription_dict['RegionName'].append(region_name)
                    prescription_dict['Date'].append(date.strftime("%Y-%m-%d"))
                    for ip in IP_COLS:
                        prescription_dict[ip].append(curr_ips[ip])

    # Create dataframe from dictionary.
    prescription_df = pd.DataFrame(prescription_dict)
    prescription_df = prescription_df.replace("",np.NaN)
    return prescription_df

def weight_prescriptions_by_cost(pres_df, cost_df):
    """
    Weight prescriptions by their costs.
    """
    weighted_df = (pres_df.merge(cost_df, how='outer', on="GeoID", suffixes=('_pres', '_cost'))
                   .rename({"CountryName_pres": "CountryName", "RegionName_pres": "RegionName"}, axis=1)
                   )
    for npi_col in NPI_COLUMNS:
        weighted_df[npi_col] = weighted_df[npi_col + '_pres'] * weighted_df[npi_col + '_cost']
    return weighted_df

def generate_cases_and_stringency_for_prescriptions(start_date, 
                                                    end_date,
                                                    prescription_df,
                                                    cost_df):
    start_time = time.time()
    # Load the prescriptions, handling Date and regions
    pres_df = XPrizePredictor.load_original_data_from_df(prescription_df)
    # print(pres_df.head())

    # Generate predictions for all prescriptions
    predictor = XPrizePredictor()
    pred_dfs = {}
    for idx in pres_df['PrescriptionIndex'].unique():
        idx_df = pres_df[pres_df['PrescriptionIndex'] == idx]
        idx_df = idx_df.drop(columns='PrescriptionIndex')  # Predictor doesn't need this
        # Generate the predictions
        pred_df = predictor.predict_from_df(start_date, end_date, idx_df)
        print(f"Generated predictions for PrescriptionIndex {idx}")
        pred_df['PrescriptionIndex'] = idx
        pred_dfs[idx] = pred_df
    pred_df = pd.concat(list(pred_dfs.values()))
    # Aggregate cases by prescription index and geo
    agg_pred_df = pred_df.groupby(['CountryName',
                                   'RegionName',
                                   'PrescriptionIndex'], dropna=False).mean().reset_index()

    # Load IP cost weights
    # Only use costs of geos we've predicted for
    
    cost_df = add_geo_id(cost_df)
    agg_pred_df = add_geo_id(agg_pred_df)
    cost_df = cost_df[cost_df.GeoID.isin(agg_pred_df.GeoID)]
    # Apply weights to prescriptions
    pres_df = weight_prescriptions_by_cost(pres_df, cost_df)
    # Aggregate stringency across npis
    pres_df['Stringency'] = pres_df[NPI_COLUMNS].sum(axis=1)

    # Aggregate stringency by prescription index and geo
    agg_pres_df = pres_df.groupby(['CountryName',
                                   'RegionName',
                                   'PrescriptionIndex'], dropna=False).mean().reset_index()
    # Combine stringency and cases into a single df
    df = agg_pres_df.merge(agg_pred_df, how='outer', on=['CountryName',
                                                         'RegionName',
                                                         'PrescriptionIndex'])
    # Only keep columns of interest
    df = df[['CountryName',
             'RegionName',
             'PrescriptionIndex',
             'PredictedDailyNewCases',
             'Stringency']]
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_tring = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print(f"Evaluated {len(pred_dfs)} PrescriptionIndex in {elapsed_time_tring} seconds")

    return df, pred_dfs
