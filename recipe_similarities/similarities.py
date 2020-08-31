"""
Use content based to weight the ratings, the cluster would work here to impute

 use a weighting system controlled by alpha parameter

 combined_similarity = alpha*content_sim+(1-alpha)*collb_sim
 alpha = 1 then content_only
 alpha = 0 then collab only
 alpha can be tuned
 http://facweb.cs.depaul.edu/mobasher/research/papers/ewmf04-web/node9.html
"""

import os
from warnings import warn

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from recipe_similarities.utils.clean_data import prepare_data
from recipe_similarities.config.defaults import raw_data_files

# BASE_DIR = os.path.dirname(os.path.realpath(__file__))
BASE_DIR = "C:/Users/Alex Georgiou/similarity_matrix/recipe_similarities"


class SimilarityFactory:
    def __init__(self,
                 recipes_info_file_path=None,
                 similarity_score_file_path=None,
                 recipe_ids=None):

        self.recipes_info_file_path = recipes_info_file_path
        self.similarity_score_file_path = similarity_score_file_path
        self.recipe_ids = recipe_ids
        self.recipes_df = None
        self.sim_scores_df = None

    def load_data(self):
        if self.recipes_info_file_path is None or self.similarity_score_file_path is None:
            warn("You have not provided a recipes.csv AND a similarity_scores.csv. Default raw data will be used. ")
            raw_data = raw_data_files()

            self.recipes_info_file_path = os.path.join(BASE_DIR,
                                                       'data',
                                                       'raw_data',
                                                       raw_data['recipes_info'])
            self.similarity_score_file_path = os.path.join(BASE_DIR,
                                                           'data',
                                                           'raw_data',
                                                           raw_data['similarity_scores'])

            self.recipes_df, self.sim_scores_df = prepare_data(self.recipes_info_file_path,
                                                               self.similarity_score_file_path)

    def get_data(self):
        return self.recipes_df

    @staticmethod
    def _prep_time_class(field):
        """
        This function converts preparation time to a class
        nb highly subjective, should ideally be validated by understanding customer perception
        """

        if field <= 20:
            return 'fast'
        elif 20 < field <= 40:
            return 'medium'
        elif field > 40:
            return 'slow'


    def _recipes_prepare_data(self):

        df = self.recipes_df.copy()

        df.index = df.recipe_id
        del df['recipe_id']
        del df['country_secondary']

        # treat missing content as information
        df.fillna('missing', inplace=True)
        df['family_friendly'].replace(to_replace={'no': 'family unfriendly',
                                                  'yes': 'family friendly'},
                                      inplace=True)

        df['dish_category'].replace(to_replace={'protein&veg': 'protein & veg'},
                                    inplace=True)
        df['prep_time'] = df['prep_time'].apply(self._prep_time_class)

        return df

    def recipes_jaccard_similarities(self):
        # utilise numpys matrix operations for fast computation
        df = self._recipes_prepare_data()

        a = df.values.copy()
        b = df.values.copy()

        all_recipes_by_n_recipes = np.repeat(a[np.newaxis, :, :],
                                             a.shape[0],
                                             axis=0)

        all_recipes = b.reshape(b.shape[0],
                                1,
                                b.shape[1])

        intersect = np.sum(all_recipes_by_n_recipes == all_recipes, axis=2)
        union = np.sum(all_recipes_by_n_recipes != all_recipes, axis=2) * 2 + intersect
        jaccard_sim = intersect / union

        jaccard_sim_df = pd.DataFrame(jaccard_sim, index=df.index, columns=df.index)

        return jaccard_sim_df

    @staticmethod
    def _concat_to_pipe_delim_str(fields):
        all_col_values = fields.tolist()

        return "||".join(all_col_values)

    @staticmethod
    def _custom_tokeniser(doc):
        return doc.split("||")

    def recipe_cosine_similarities(self):
        df = self._recipes_prepare_data()
        df['delim_str'] = df.apply(self._concat_to_pipe_delim_str, axis=1)
        tf_idf = TfidfVectorizer(analyzer='word',
                                 min_df=0,
                                 tokenizer=self._custom_tokeniser
                                 )
        tfidf_matrix = tf_idf.fit_transform(df['delim_str'])
        cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        cos_sim_df = pd.DataFrame(cos_sim, index=df.index, columns=df.index)

        return cos_sim_df

    def scored_cosine_similarities(self):
        df = self.sim_scores_df
        edges = df.groupby(['recipe_a', 'recipe_b'])\
                  .agg('mean')\
                  .reset_index()[['recipe_a', 'recipe_b', 'score']]
        graph = nx.from_pandas_dataframe(edges, 'recipe_a', 'recipe_b', 'score')
        adj_matrix = nx.to_numpy_matrix(graph, weight='score')
        nodes = [int(n) for n in graph.nodes()]
        adj_matrix_df = pd.DataFrame(adj_matrix, index=nodes, columns=nodes)
        cos_sim = cosine_similarity(adj_matrix_df, adj_matrix_df)
        cos_sim_df = pd.DataFrame(cos_sim, index=adj_matrix_df.index, columns=adj_matrix_df.index)
        return cos_sim_df

    def _join_weights_to_recipes(self):
        pass

    def hybrid_weighted_similarity(self):
        pass


sf = SimilarityFactory()
sf.load_data()
cb_jaccard = sf.recipes_jaccard_similarities()
cb_cos = sf.recipe_cosine_similarities()
cbf_cos = sf.scored_cosine_similarities()
print(cb_jaccard.shape, cb_cos.shape, cbf_cos.shape)


def create_master_index(all_matrices):
    all_recipe_ids = [list(li.index) for li in all_matrices]
    union = set().union(*all_recipe_ids)

    return sorted(list(union))


all_m = [cb_jaccard, cb_cos, cbf_cos]

all_ids = create_master_index(all_m)
len(all_ids)

blank_matrix = np.where(np.zeros((len(all_ids),len(all_ids)))==0, np.nan, np.nan)

all_recipes_cb = pd.DataFrame(blank_matrix.copy(), index=all_ids, columns=all_ids)
all_recipes_cb.loc[cb_cos.index,cb_cos.columns] = cb_jaccard.copy()

all_recipes_cbf = pd.DataFrame(blank_matrix.copy(), index=all_ids, columns=all_ids)
all_recipes_cbf.loc[cbf_cos.index,cbf_cos.columns] = cbf_cos.copy()

alpha = 0.9

cbf_mask = all_recipes_cbf.isnull().replace(to_replace={True:0.0, False:alpha})
cbf_weighted = (cbf_mask*all_recipes_cbf).fillna(0)

cb_mask = all_recipes_cb.isnull().replace(to_replace={True:0.0, False:1.0-alpha})
cb_weighted = (cb_mask*all_recipes_cb).fillna(0)

hybrid_sim = cb_weighted+cbf_weighted
id = 57
top4 = hybrid_sim.loc[id].sort_values(ascending=False)[0:5]
top4.name = 'similarity_score'

df = pd.read_csv('recipe_similarities/data/clean_data/recipes_info.csv')

df.index=df['recipe_id']

top4_df = df.loc[top4.index].copy()
top4_df = top4_df.merge(top4, left_index=True, right_index=True, how='left')
print(top4_df.T)