import pandas as pd
import dgl
import os
import torch
import numpy as np
import scipy.sparse as sp
import time
from functools import partial
from .. import randomwalk
import stanfordnlp
import re
import tqdm
import string
from .base import UserProductDataset

class MovieLens(UserProductDataset):
    def __init__(self, directory):
        '''
        directory: path to movielens directory which should have the three
                   files:
                   users.dat
                   products.dat
                   ratings.dat
        '''
        self.directory = directory

        users = []
        products = []
        ratings = []

        # read users
        with open(os.path.join(directory, 'users.dat')) as f:
            for l in f:
                id_, gender, age, occupation, zip_ = l.strip().split('::')
                users.append({
                    'id': int(id_),
                    'gender': gender,
                    'age': age,
                    'occupation': occupation,
                    'zip': zip_,
                    })
        self.users = pd.DataFrame(users).set_index('id').astype('category')

        # read products
        with open(os.path.join(directory, 'movies.dat'), encoding='latin1') as f:
            for l in f:
                id_, title, genres = l.strip().split('::')
                genres_set = set(genres.split('|'))

                # extract year
                assert re.match(r'.*\([0-9]{4}\)$', title)
                year = title[-5:-1]
                title = title[:-6].strip()

                data = {'id': int(id_), 'title': title, 'year': year}
                for g in genres_set:
                    data[g] = True
                products.append(data)
        self.products = (
                pd.DataFrame(products)
                .set_index('id')
                .fillna(False)
                .astype({'year': 'category'}))
        self.genres = self.products.columns[self.products.dtypes == bool]

        # read ratings
        with open(os.path.join(directory, 'ratings.dat')) as f:
            for l in f:
                user_id, product_id, rating, timestamp = [int(_) for _ in l.split('::')]
                ratings.append({
                    'user_id': user_id,
                    'product_id': product_id,
                    'rating': rating,
                    'timestamp': timestamp,
                    })
        ratings = pd.DataFrame(ratings)
        product_count = ratings['product_id'].value_counts()
        product_count.name = 'product_count'
        ratings = ratings.join(product_count, on='product_id')
        self.ratings = ratings

        # drop users and products which do not exist in ratings
        self.users = self.users[self.users.index.isin(self.ratings['user_id'])]
        self.products = self.products[self.products.index.isin(self.ratings['product_id'])]

        self.ratings = self.data_split(self.ratings)
        self.build_graph()
        self.find_neighbors(0.2, 2000, 1000)

    def build_graph(self):
        user_ids = list(self.users.index)
        product_ids = list(self.products.index)
        user_ids_invmap = {id_: i for i, id_ in enumerate(user_ids)}
        product_ids_invmap = {id_: i for i, id_ in enumerate(product_ids)}
        self.user_ids = user_ids
        self.product_ids = product_ids
        self.user_ids_invmap = user_ids_invmap
        self.product_ids_invmap = product_ids_invmap

        g = dgl.DGLGraph()
        g.add_nodes(len(user_ids) + len(product_ids))

        # user features
        for user_column in self.users.columns:
            udata = torch.zeros(g.number_of_nodes(), dtype=torch.int64)
            # 0 for padding
            udata[:len(user_ids)] = \
                    torch.LongTensor(self.users[user_column].cat.codes.values.astype('int64') + 1)
            g.ndata[user_column] = udata

        # product genre
        product_genres = torch.from_numpy(self.products[self.genres].values.astype('float32'))
        g.ndata['genre'] = torch.zeros(g.number_of_nodes(), len(self.genres))
        g.ndata['genre'][len(user_ids):len(user_ids) + len(product_ids)] = product_genres

        # product year
        g.ndata['year'] = torch.zeros(g.number_of_nodes(), dtype=torch.int64)
        # 0 for padding
        g.ndata['year'][len(user_ids):len(user_ids) + len(product_ids)] = \
                torch.LongTensor(self.products['year'].cat.codes.values.astype('int64') + 1)

        # product title
        nlp = stanfordnlp.Pipeline(use_gpu=False, processors='tokenize,lemma')
        vocab = set()
        title_words = []
        for t in tqdm.tqdm(self.products['title'].values):
            doc = nlp(t)
            words = set()
            for s in doc.sentences:
                words.update(w.lemma.lower() for w in s.words
                             if not re.fullmatch(r'['+string.punctuation+']+', w.lemma))
            vocab.update(words)
            title_words.append(words)
        vocab = list(vocab)
        vocab_invmap = {w: i for i, w in enumerate(vocab)}
        # bag-of-words
        g.ndata['title'] = torch.zeros(g.number_of_nodes(), len(vocab))
        for i, tw in enumerate(tqdm.tqdm(title_words)):
            g.ndata['title'][i, [vocab_invmap[w] for w in tw]] = 1
        self.vocab = vocab
        self.vocab_invmap = vocab_invmap

        rating_user_vertices = [user_ids_invmap[id_] for id_ in self.ratings['user_id'].values]
        rating_product_vertices = [product_ids_invmap[id_] + len(user_ids)
                                 for id_ in self.ratings['product_id'].values]
        self.rating_user_vertices = rating_user_vertices
        self.rating_product_vertices = rating_product_vertices

        g.add_edges(
                rating_user_vertices,
                rating_product_vertices,
                data={'inv': torch.zeros(self.ratings.shape[0], dtype=torch.uint8)})
        g.add_edges(
                rating_product_vertices,
                rating_user_vertices,
                data={'inv': torch.ones(self.ratings.shape[0], dtype=torch.uint8)})
        self.g = g

    def find_neighbors(self, restart_prob, max_nodes, top_T):
        # TODO: replace with more efficient PPR estimation
        neighbor_probs, neighbors = randomwalk.random_walk_distribution_topt(
                self.g, self.g.nodes(), restart_prob, max_nodes, top_T)

        self.user_neighbors = []
        for i in range(len(self.user_ids)):
            user_neighbor = neighbors[i]
            self.user_neighbors.append(user_neighbor.tolist())

        self.product_neighbors = []
        for i in range(len(self.user_ids), len(self.user_ids) + len(self.product_ids)):
            product_neighbor = neighbors[i]
            self.product_neighbors.append(product_neighbor.tolist())

    def generate_mask(self):
        while True:
            ratings = self.ratings.groupby('user_id', group_keys=False).apply(self.split_user)
            prior_prob = ratings['prob'].values
            for i in range(5):
                train_mask = (prior_prob >= 0.2 * i) & (prior_prob < 0.2 * (i + 1))
                prior_mask = ~train_mask
                train_mask &= ratings['train'].values
                prior_mask &= ratings['train'].values
                yield prior_mask, train_mask

    def refresh_mask(self):
        if not hasattr(self, 'masks'):
            self.masks = self.generate_mask()
        prior_mask, train_mask = next(self.masks)

        valid_tensor = torch.from_numpy(self.ratings['valid'].values.astype('uint8'))
        test_tensor = torch.from_numpy(self.ratings['test'].values.astype('uint8'))
        train_tensor = torch.from_numpy(train_mask.astype('uint8'))
        prior_tensor = torch.from_numpy(prior_mask.astype('uint8'))
        edge_data = {
                'prior': prior_tensor,
                'valid': valid_tensor,
                'test': test_tensor,
                'train': train_tensor,
                }

        self.g.edges[self.rating_user_vertices, self.rating_product_vertices].data.update(edge_data)
        self.g.edges[self.rating_product_vertices, self.rating_user_vertices].data.update(edge_data)