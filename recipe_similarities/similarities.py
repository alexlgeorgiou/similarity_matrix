import os
from warnings import warn

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from recipe_similarities.utils.clean_data import prepare_data
from recipe_similarities.config.defaults import raw_data_files

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


class SimilarityFactory:
    """
    This class produces a hybrid similarity class. To enable it to use non-example data prodided
    as a part of the exercise, you must specify the full explcit file paths in
    recipes_info_file_path and similarity_score_file_path upon intiating the class.

    Basic Usage:
    from recipe_similarities.similarities import SimilarityFactory
    sf = SimilarityFactory()
    sf.load_data()
    hsims = sf.hybrid_similarities()

    :returns dict {'similarity_matrix': NxN matics as a numpy array
                  , 'index': Pandas Index object highlightin recipe IDs to cross reference with the similarity matrix}

    """
    def __init__(self,
                 recipes_info_file_path=None,
                 similarity_score_file_path=None):

        self.recipes_info_file_path = recipes_info_file_path
        self.similarity_score_file_path = similarity_score_file_path
        self.recipes_df = None
        self.sim_scores_df = None

    def load_data(self):
        """ This function loads example data or that specifed in the path specified on class instantiation """
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
        """ This function transforms recipe data for content based methods """
        df = self.recipes_df.copy()
        # enables simple indexing of recipes throughout processes
        df.index = df.recipe_id

        del df['recipe_id']
        # removed as often duplicate info
        del df['country_secondary']

        # treat missing content as information
        df.fillna('missing', inplace=True)

        # update lalling for consitency and to simplfy string vectoriations
        df['family_friendly'].replace(to_replace={'no': 'family unfriendly',
                                                  'yes': 'family friendly'},
                                      inplace=True)

        df['dish_category'].replace(to_replace={'protein&veg': 'protein & veg'},
                                    inplace=True)

        # simplfies prep time into a categry with lower cardinality
        df['prep_time'] = df['prep_time'].apply(self._prep_time_class)

        return df

    def recipes_jaccard_similarities(self):
        """ This function computes Jaccards similarity
             uses numpy's matrix operations for fast computation of jaccards similarity"""

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
    def _concat_to_pipe_delim_str(fields, delim="||"):
        """ This function concatenates all fields into a sigle delimted column tokenisation """
        all_col_values = fields.tolist()

        return delim.join(all_col_values)

    @staticmethod
    def _custom_tokeniser(doc, delim="||"):
        """ This function is a customer tokeniser to maintain each fields whole ngram"""
        return doc.split(delim)

    def recipe_cosine_similarities(self, strict=True, vectorise_method='tfidf'):
        """
        :param strict=True default. This enforces the tokenisation where each fields value is the token.
                      False. This uses a default tokeniser to compare each recipes as if it were a single text blob.
        "param vectorise_method='tfidf' uses the tfidf document vecorisation method. Also accepts 'count'
                                'count' counts when a string is present in both recipes
        """
        df = self._recipes_prepare_data()
        df['delim_str'] = df.apply(self._concat_to_pipe_delim_str, axis=1)

        if strict:
            tokeniser = self._custom_tokeniser
            df['delim_str'] = df.apply(self._concat_to_pipe_delim_str, axis=1)

        else:
            # sets tokeniser to default english language
            tokeniser = None
            df['delim_str'] = df.apply(self._concat_to_pipe_delim_str,  delim=" ", axis=1)

        if vectorise_method == 'tfidf':
            vectoriser = TfidfVectorizer(analyzer='word',
                                         min_df=0,
                                         tokenizer=tokeniser
                                         )

        elif vectorise_method == 'count':
            vectoriser = CountVectorizer(analyzer='word',
                                         min_df=0,
                                         tokenizer=tokeniser
                                         )

        vectorised_matrix = vectoriser.fit_transform(df['delim_str'])
        cos_sim = cosine_similarity(vectorised_matrix, vectorised_matrix)
        cos_sim_df = pd.DataFrame(cos_sim, index=df.index, columns=df.index)

        return cos_sim_df

    def scored_cosine_similarities(self):
        """ :returns cosine similarity matrix of the scored recipes only"""
        df = self.sim_scores_df
        edges = df.groupby(['recipe_a', 'recipe_b'])\
                  .agg('mean')\
                  .reset_index()[['recipe_a', 'recipe_b', 'score']]
        # use of network x simplifies the computation of an undirected adjacency matrix
        # this is as a recipe can exist in one column but not the other
        graph = nx.convert_matrix.from_pandas_edgelist(edges, 'recipe_a', 'recipe_b', 'score')
        adj_matrix = nx.to_numpy_matrix(graph, weight='score')
        nodes = [int(n) for n in graph.nodes()]
        adj_matrix_df = pd.DataFrame(adj_matrix, index=nodes, columns=nodes)
        cos_sim = cosine_similarity(adj_matrix_df, adj_matrix_df)
        cos_sim_df = pd.DataFrame(cos_sim, index=adj_matrix_df.index, columns=adj_matrix_df.index)
        return cos_sim_df

    @staticmethod
    def _create_master_index(all_matrices):
        """ This function creates a deduplicated and sorted list of recipe ids across multiple matrices"""
        all_recipe_ids = [list(li.index) for li in all_matrices]
        union = set().union(*all_recipe_ids)

        return sorted(list(union))

    def hybrid_similarities(self, 
                            alpha=0.9, 
                            cb_method='cosine',
                            cbf_method='cosine', 
                            text_vectoriser='tfidf', 
                            strict=True):

        """

        :param alpha: this adjusts the weighting 1 = pure scored similarities only,
                                                 0 = pure content similarities only,
                                                 In between 0 - 1 weights either side
        :param cb_method: accepts 'cosine' or 'jaccard'
        :param cbf_method: accepts 'cosine' only
        :param text_vectoriser: accepts 'tfid' or 'count'
        :param strict: True default. This enforces the tokenisation where each field's whole string is a token.
                      False. This uses a default tokeniser to compare each recipes as if it were a single text blob.
        :return: :returns dict {'similarity_matrix': NxN matics as a numpy array
                  , 'index': Pandas Index object highlightin recipe IDs to cross reference with the similarity matrix}
        """
        
        if cb_method == 'cosine':
            cb = self.recipe_cosine_similarities(strict=True, vectorise_method=text_vectoriser)
        elif cb_method == 'jaccard':
            cb = self.recipes_jaccard_similarities()

        if cbf_method == 'cosine':
            cbf = self.scored_cosine_similarities()
 
        all_m = [cb, cbf]
        all_ids = self._create_master_index(all_m)

        blank_matrix = np.where(np.zeros((len(all_ids), len(all_ids))) == 0, np.nan, np.nan)

        cb_all_recipes = pd.DataFrame(blank_matrix.copy(), index=all_ids, columns=all_ids)
        cb_all_recipes.loc[cb.index, cb.columns] = cb.copy()
        cb_mask = cb_all_recipes.isnull().replace(to_replace={True: 0.0, False: 1.0 - alpha})
        cb_weighted = (cb_mask * cb_all_recipes).fillna(0)

        cbf_all_recipes = pd.DataFrame(blank_matrix.copy(), index=all_ids, columns=all_ids)
        cbf_all_recipes.loc[cbf.index, cbf.columns] = cbf.copy()
        cbf_mask = cbf_all_recipes.isnull().replace(to_replace={True: 0.0, False: alpha})
        cbf_weighted = (cbf_mask * cbf_all_recipes).fillna(0)

        hybrid_sim = cb_weighted + cbf_weighted

        return {'similarity_matrix': hybrid_sim, 'index': hybrid_sim.index}
