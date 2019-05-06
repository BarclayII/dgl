import dgl
import numpy as np
import pickle
import pandas as pd
import os
import tqdm
from functools import partial
from .base import UserProductDataset

def get_token_list(df, old_field, new_field):
    df[new_field] = (
            df[old_field].str
            .split(',')
            .map(lambda x: [(int(i) + 1) if i != '' else 0 for i in x]))
    max_length = df[new_field].map(len).max()
    df[new_field] = df[new_field].map(lambda x: x + [0] * (max_length - len(x)))

def merge_users(session_user_mapping):
    session_user_mapping = session_user_mapping.dropna(subset=['userId'])
    users_merged = session_user_mapping.groupby('sessionId')['userId'].unique()
    users = session_user_mapping['userId'].unique().astype('int64')

    # union-find set to merge all the users
    parent = np.arange(len(users))
    reverse_mapping = {u: i for i, u in enumerate(users)}

    def find(parent, u):
        if u == parent[u]:
            return u
        parent[u] = find(parent, parent[u])
        return parent[u]

    def union(parent, u, v):
        if find(parent, u) == find(parent, v):
            return
        parent[u] = v

    for user_set in users_merged:
        for i in range(len(user_set) - 1):
            union(parent, reverse_mapping[user_set[i]], reverse_mapping[user_set[i + 1]])

    user_mapping = {u: users[find(parent, reverse_mapping[u])] for u in users}
    user_mapping[np.nan] = np.nan
    return user_mapping


def fill_users(user_mapping, table):
    table = table.fillna({'userId': table.groupby('sessionId')['userId'].transform('first')})
    table['userId'] = table['userId'].map(user_mapping)
    return table


class CIKM(UserProductDataset):
    def __init__(self, path):
        super(CIKM, self).__init__()

        self.path = path

        train_clicks = pd.read_csv(os.path.join(path, 'train-clicks.csv'), sep=';')
        train_purchases = pd.read_csv(os.path.join(path, 'train-purchases.csv'), sep=';')
        train_item_views = pd.read_csv(os.path.join(path, 'train-item-views.csv'), sep=';')
        all_queries = pd.read_csv(os.path.join(path, 'train-queries.csv'), sep=';')
        products = pd.read_csv(os.path.join(path, 'products.csv'), sep=';')
        product_categories = pd.read_csv(os.path.join(path, 'product-categories.csv'), sep=';')
        test_queries = all_queries[all_queries['is.test']].drop('is.test', axis=1)
        train_queries = all_queries[~all_queries['is.test']].drop('is.test', axis=1)

        # This dataset only take the following features:
        # User id
        # Query clicks (edge)
        # Views (edge)
        # Purchases (edge) (treated the same as views)
        # Product description
        # Product category

        # merge users that ever appeared in the same session
        session_user_mapping = pd.concat([
            train_purchases[['sessionId', 'userId']],
            train_item_views[['sessionId', 'userId']],
            train_queries[['sessionId', 'userId']],
            ])
        user_mapping = merge_users(session_user_mapping)

        train_queries_with_clicks = train_queries.join(train_clicks.set_index('queryId')['itemId'], on='queryId', how='inner')

        train_queries_with_clicks = fill_users(user_mapping, train_queries_with_clicks)
        train_purchases = fill_users(user_mapping, train_purchases)
        train_item_views = fill_users(user_mapping, train_item_views)
        test_queries = fill_users(user_mapping, test_queries)

        ratings_with_session = pd.concat([
            train_queries_with_clicks[['sessionId', 'userId', 'itemId', 'searchstring.tokens', 'categoryId']],
            train_item_views[['sessionId', 'userId', 'itemId']],
            train_purchases[['sessionId', 'userId', 'itemId']],
            ]).drop_duplicates()
        ratings_with_session = (
                ratings_with_session
                .fillna({'categoryId': 0, 'searchstring.tokens': ''})
                .astype({'categoryId': 'int64'}))
        ratings = ratings_with_session.drop('sessionId', axis=1).drop_duplicates()
        ratings = ratings.dropna(subset=['userId'])
        get_token_list(ratings_with_session, 'searchstring.tokens', 'tokens')
        get_token_list(ratings, 'searchstring.tokens', 'tokens')

        get_token_list(train_queries_with_clicks, 'items', 'itemList')
        query_candidates = []
        for index, row in tqdm.tqdm(train_queries_with_clicks.iterrows()):
            query_candidates.extend((row['queryId'], item) for item in row['itemList'])
        self.query_candidates = pd.DataFrame(columns=['query_id', 'product_id'], data=query_candidates).drop_duplicates()

        self.users = pd.DataFrame({'id': ratings['userId'].unique()}).set_index('id')
        self.products = pd.DataFrame(
                {'itemId': list(set(ratings['itemId']) | set(self.query_candidates['product_id']))}
                ).set_index('itemId')
        self.products = (
                self.products
                .join(products.set_index('itemId'), on='itemId', how='left')
                .fillna({'product.name.tokens': '', 'pricelog2': products['pricelog2'].mean()})
                .join(product_categories.set_index('itemId'), on='itemId', how='left')
                .fillna({'categoryId': 0}))
        self.query_ids = self.query_candidates['query_id'].unique()
        assert self.products['product.name.tokens'].notnull().all()
        assert self.products['categoryId'].notnull().all()

        get_token_list(self.products, 'product.name.tokens', 'tokens')

        ratings = ratings.rename({'userId': 'user_id', 'itemId': 'product_id'}, axis=1)
        ratings_with_session = ratings_with_session.rename({'userId': 'user_id', 'itemId': 'product_id'}, axis=1)
        self.ratings_with_session = ratings_with_session
        self.ratings_complete = ratings

        product_count = ratings['product_id'].value_counts()
        user_count = ratings['user_id'].value_counts()
        product_count.name = 'product_count'
        user_count.name = 'user_count'
        ratings = ratings.join(product_count, on='product_id').join(user_count, on='user_id')
        self.ratings = ratings

        self.ratings = self.data_split(self.ratings)

        self.build_graph()
        self.test_queries = test_queries
        self.train_queries_with_clicks = train_queries_with_clicks
        self.train_purchases = train_purchases
        self.train_item_views = train_item_views
        self.anonymous_ratings = self.ratings_with_session[self.ratings_with_session['user_id'].isnull()]

    def build_graph(self):
        # The graph consists of user nodes, item nodes, and query nodes.
        # User and items form a bipartite graph while item and queries form another one.
        # Note: we always perform 2-step random walks, so the neighbors of an item would
        # always be an item.
        # Note: the embedding of query nodes are NOT learned or used.
        import torch
        user_ids = list(self.users.index)
        product_ids = list(self.products.index)
        query_ids = self.query_ids
        user_ids_invmap = {id_: i for i, id_ in enumerate(user_ids)}
        product_ids_invmap = {id_: i for i, id_ in enumerate(product_ids)}
        query_ids_invmap = {id_: i for i, id_ in enumerate(query_ids)}
        self.user_ids = user_ids
        self.product_ids = product_ids
        self.user_ids_invmap = user_ids_invmap
        self.product_ids_invmap = product_ids_invmap

        g = dgl.DGLGraph(multigraph=True)
        g.add_nodes(len(user_ids) + len(product_ids) + len(query_ids))

        tokens = torch.LongTensor(np.array(self.products['tokens'].tolist()))
        tokens = torch.cat(
                [torch.zeros(len(user_ids), tokens.shape[1], dtype=torch.int64),
                 tokens,
                 torch.zeros(len(query_ids), tokens.shape[1], dtype=torch.int64)],
                0)
        logprice = torch.FloatTensor(self.products['pricelog2'].to_numpy())
        logprice = torch.cat(
                [torch.zeros(len(user_ids)), logprice, torch.zeros(len(query_ids))],
                0)
        category = torch.LongTensor(self.products['categoryId'].to_numpy())
        category = torch.cat(
                [torch.zeros(len(user_ids), dtype=torch.int64),
                 category,
                 torch.zeros(len(query_ids), dtype=torch.int64)],
                0)
        g.ndata['tokens'] = tokens
        g.ndata['nid'] = torch.cat([
            torch.arange(1, 1 + len(user_ids)),
            torch.zeros(len(product_ids) + len(query_ids), dtype=torch.int64)])
        g.ndata['category'] = category
        g.ndata['logprice'] = logprice[:, None]

        rating_user_vertices = [user_ids_invmap[id_] for id_ in self.ratings['user_id'].values]
        rating_product_vertices = [product_ids_invmap[id_] + len(user_ids)
                                 for id_ in self.ratings['product_id'].values]
        self.rating_user_vertices = rating_user_vertices
        self.rating_product_vertices = rating_product_vertices
        tokens = torch.LongTensor(np.array(self.ratings['tokens'].tolist()))
        tokens = torch.cat([torch.zeros(1, tokens.shape[1]).long(), tokens], 0)
        category = torch.LongTensor(self.ratings['categoryId'].to_numpy())
        self.query_tokens = tokens

        # NOTE: 'tokens' and 'category' do not take part in computation of
        # user/item node embeddings; they join after PinSage computation
        g.add_edges(
                rating_user_vertices,
                rating_product_vertices,
                data={
                    'inv': torch.zeros(self.ratings.shape[0], dtype=torch.uint8),
                    'tokens_idx': torch.arange(1, len(rating_user_vertices) + 1).long(),
                    'category': category,
                    })
        g.add_edges(
                rating_product_vertices,
                rating_user_vertices,
                data={
                    'inv': torch.ones(self.ratings.shape[0], dtype=torch.uint8),
                    'tokens_idx': torch.arange(1, len(rating_user_vertices) + 1).long(),
                    'category': category,
                    })
        cand_query_vertices = [query_ids_invmap[id_] + len(user_ids) + len(product_ids)
                for id_ in self.query_candidates['query_id'].values]
        cand_product_vertices = [product_ids_invmap[id_] + len(user_ids)
                for id_ in self.query_candidates['product_id'].values]
        g.add_edges(cand_query_vertices, cand_product_vertices)
        g.add_edges(cand_product_vertices, cand_query_vertices)

        self.g = g
        self.g.readonly()

    def generate_mask(self):
        ratings = self.ratings
        while True:
            ratings_grouped = ratings.groupby('user_id').apply(
                    partial(self.split_user_product, filter_counts=5))
            prior_prob = ratings_grouped['prob'].values
            for i in range(5):
                train_mask = (prior_prob >= 0.2 * i) & (prior_prob < 0.2 * (i + 1))
                prior_mask = ~train_mask
                train_mask &= ratings_grouped['train'].values
                prior_mask &= ratings_grouped['train'].values
                yield prior_mask, train_mask

    def split_user_product(self, df, filter_counts=0):
        df_new = df.copy()
        df_new['prob'] = -1

        df_new_sub = (
                (df_new['product_count'] >= filter_counts) |
                (df_new['user_count'] >= filter_counts)
                ).nonzero()[0]
        prob = np.linspace(0, 1, df_new_sub.shape[0], endpoint=False)
        np.random.shuffle(prob)
        df_new['prob'].iloc[df_new_sub] = prob
        return df_new
