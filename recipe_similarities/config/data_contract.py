import numpy as np


def raw_data_contract():

    return {'similarity_scores': {'file_format': 'csv',
                                  'columns': {'user_id': np.dtype(np.int64),
                                              'message_ts': np.dtype(np.object),
                                              'action_ts': np.dtype(np.object),
                                              'recipe_a': np.dtype(np.int64),
                                              'recipe_b': np.dtype(np.int64),
                                              'score': np.dtype(np.int64)},
                                  'pct_acceptable_nulls': 0,
                                  'min_max': {'score': [1, 4]}
                                  },
            'recipes_info': {'file_format': 'csv',
                             'columns': {'carbohydrate_base': np.dtype(np.object),
                                         'carbohydrate_category': np.dtype(np.object),
                                         'country': np.dtype(np.object),
                                         'country_secondary': np.dtype(np.object),
                                         'diet_type': np.dtype(np.object),
                                         'dish_category': np.dtype(np.object),
                                         'dish_type': np.dtype(np.object),
                                         'family_friendly': np.dtype(np.object),
                                         'prep_time': np.dtype(np.object),
                                         'protein': np.dtype(np.object),
                                         'protein_cut': np.dtype(np.object),
                                         'protein_type': np.dtype(np.object),
                                         'recipe_id': np.dtype(np.int64),
                                         'spice_level': np.dtype(np.object)},
                             'pct_acceptable_nulls': 0.5,
                             'min_max': None
                             }
            }

