import os
import re

import numpy as np
import pandas as pd

from recipe_similarities.config.data_contract import raw_data_contract
from recipe_similarities.config.defaults import Config

DATA_DIR = os.path.join(os.getcwd(), 'data')


def validate_min_max(expected, df):

    for col, accept_range in expected.items():
        assert df[col].min() >= accept_range[0]
        assert df[col].max() >= accept_range[1]


def validate_raw_data(contract, df):

    # check column schema is expected
    assert dict(df.dtypes) == contract['columns']

    # check there are nulls below a configured level
    assert df.isnull().sum().sum()/(df.shape[0]*df.shape[1]) <= contract['pct_acceptable_nulls']

    if contract['min_max']:
        validate_min_max(contract['min_max'], df)


def safe_lower(field):
    if type(field) == str:
        return field.lower()
    else:
        return field


def max_prep_time(field):
    match_rule = '[0-9]{1,3}-[0-9]{1,3}'
    if re.match(match_rule, field):
        time_range = [int(v) for v in field.split('-')]

        return max(time_range)
    else:
        return int(field)


def recipe_data_prep(df):

    # standardise all string capitalisation
    columns_to_lower = ['carbohydrate_base',
                        'carbohydrate_category',
                        'country',
                        'country_secondary',
                        'diet_type',
                        'dish_category',
                        'dish_type',
                        'family_friendly',
                        'prep_time',
                        'protein',
                        'protein_cut',
                        'protein_type',
                        'spice_level']

    for col in columns_to_lower:
        df[col] = df[col].apply(safe_lower)

    # replace duplicated strings
    df['carbohydrate_base'].replace(to_replace={'carb not found': np.nan},
                                    inplace=True)
    df['carbohydrate_category'].replace(to_replace={'carb not found':  np.nan},
                                        inplace=True)

    df['country'].replace(to_replace={'great britain': 'united kingdom',
                                      'korea, republic of (south korea)': 'south korea',
                                      'israel and the occupied territories': 'israel'},
                          inplace=True)
    df['country_secondary'].replace(to_replace={'great britain': 'united kingdom',
                                                'korea, republic of (south korea)': 'south korea',
                                                'israel and the occupied territories': 'israel'},
                                    inplace=True)

    # take the max cook time
    df['prep_time'] = df['prep_time'].apply(max_prep_time)

    return df


def save_clean_data(df, file_name):
    df.to_csv(os.path.join(DATA_DIR, 'clean_data', file_name), index=False)


def read_raw_data(file_name):
    fpath = os.path.join(DATA_DIR,
                         'raw_data',
                         file_name)
    df = pd.read_csv(fpath)

    return df


def run():

    raw_data_files = Config.raw_data_files()
    contract = raw_data_contract()

    # clean and save similarty scores to clean data
    sim_scores_df = read_raw_data(raw_data_files['similarity_scores'])

    validate_raw_data(contract['similarity_scores'], sim_scores_df)

    save_clean_data(sim_scores_df, raw_data_files['similarity_scores'])

    # clean and save recipe info to clean data
    recipe_info_df = read_raw_data(raw_data_files['recipes_info'])

    validate_raw_data(contract['recipes_info'], recipe_info_df)

    clean_recipe_info_df = recipe_data_prep(recipe_info_df)
    print(clean_recipe_info_df)
    save_clean_data(clean_recipe_info_df, raw_data_files['recipes_info'])

run()