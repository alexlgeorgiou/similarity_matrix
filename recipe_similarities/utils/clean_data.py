import os
import re

import numpy as np
import pandas as pd

from recipe_similarities.config.data_contract import raw_data_contract


def validate_min_max(expected, df):

    for col, accept_range in expected.items():
        assert df[col].min() >= accept_range[0]
        assert df[col].max() <= accept_range[1]


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

    # replace duplicated/ conflicting strings
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

    # some duplicate recipes exist drop them
    df.drop_duplicates(subset='recipe_id', inplace=True)

    return df


def prepare_data(recipes_info_file_path,
                 similarity_score_file_path):

    contract = raw_data_contract()

    sim_scores_df = pd.read_csv(similarity_score_file_path)
    validate_raw_data(contract['similarity_scores'], sim_scores_df)

    recipe_info_df = pd.read_csv(recipes_info_file_path)
    validate_raw_data(contract['recipes_info'], recipe_info_df)
    clean_recipe_info_df = recipe_data_prep(recipe_info_df)

    return [clean_recipe_info_df, sim_scores_df]
