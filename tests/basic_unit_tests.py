import pandas as pd
import pytest

from recipe_similarities.utils.clean_data import validate_raw_data
from recipe_similarities.config.data_contract import raw_data_contract
from recipe_similarities.similarities import SimilarityFactory


def test_data_validator_error():
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

    contract = raw_data_contract()
    with pytest.raises(KeyError):
        validate_raw_data(contract['recipes_info'], df)


def test_data_validator():

    example_data = {'action_ts': {0: '28/07/2016 16:32',
                                  1: '11/08/2016 12:08',
                                  2: '15/08/2016 11:11',
                                  3: '20/08/2016 22:43',
                                  4: '25/08/2016 11:27'},
                    'message_ts': {0: '28/07/2016 15:22',
                                    1: '11/08/2016 12:06',
                                    2: '15/08/2016 11:04',
                                    3: '20/08/2016 22:42',
                                    4: '25/08/2016 10:38'},
                    'recipe_a': {0: 548, 1: 584, 2: 288, 3: 271, 4: 585},
                    'recipe_b': {0: 292, 1: 288, 2: 553, 3: 498, 4: 89},
                    'score': {0: 2, 1: 1, 2: 3, 3: 2, 4: 1},
                    'user_id': {0: 10141, 1: 10163, 2: 10163, 3: 10011, 4: 10141}}

    df = pd.DataFrame(example_data)
    contract = raw_data_contract()
    validate_raw_data(contract['similarity_scores'], df)


def test_defaults():
    sf = SimilarityFactory()
    sf.load_data()
    hs = sf.hybrid_similarities(alpha=0.5)
    assert hs['similarity_matrix'].shape == (262, 262)
    assert round(hs['similarity_matrix'].sum().sum()) == 5286
    assert hs['index'].shape == (262,)
