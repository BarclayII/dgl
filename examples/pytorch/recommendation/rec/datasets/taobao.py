import dgl
import pandas as pd
import json
import os
import numpy as np

class Taobao(UserProductDataset):
    def __init__(self, directory):
        self.directory = directory
        ratings = []
        user_id = 0

        with open(os.path.join(directory, 'test/all.json')) as f:
            for l in f:
                dic = json.loads(l)
                item_ids, masks = dic['item_ids'], dic['masks']
                ratings.extend([{
                    'user_id': user_id,
                    'product_id': item_id,
                    'prior': mask == 0,
                    'train': False,
                    'valid': False,
                    'test': mask == 1}
                    for item_id, mask in zip(item_ids, masks)])
        with open(os.path.join(directory, 'train/all.json')) as f:
            for l in f:
                dic = json.loads(l)
                item_ids, masks = dic['item_ids'], dic['masks']
                masks[-5:] = 2
                ratings.extend([{
                    'user_id': user_id,
                    'product_id': item_id,
                    'prior': mask == 0,
                    'train': mask == 1,
                    'valid': mask == 2,
                    'test': False}])

        ratings = pd.DataFrame(ratings)
        product_count = ratings['product_id'].value_counts()
        product_coount.name = 'product_count'
        ratings = ratings.join(product_count, on='product_id')
        self.ratings = ratings

        self.users = pd.Series(ratings['user_id'].unique()).to_frame().set_index('user_id')
        self.products = pd.Series(ratings['product_id'].unique()).to_frame().set_index('product_id')
        self.build_graph()
        self.generate_candidates()

    def build_graph(self):
        user_ids = list(self.users.index)
        product_ids = list(self.products.index)
        user_ids_invmap = {id_: i for i, id_ in enumerate(user_ids)}
        product_ids_invmap = {id_: i for i, id_ in enumerate(product_ids)}
        self.user_ids = user_ids
        self.product_ids = product_ids
        self.user_ids_invmap = user_ids_invmap
        self.product_ids_invmap = product_ids_invmap

        g = dgl.DGLGraph(multigraph=True)
        g.add_nodes(len(user_ids) + len(product_ids))
        rating_user_vertices = [user_ids_invmap[id_] for id_ in ratings['user_id'].values]
        rating_product_vertices = [product_ids_invmap[id_] for id_ in ratings['product_id'].values]
        self.rating_user_vertices = rating_user_vertices
        self.rating_product_vertices = rating_product_vertices

        g.add_edges(
                rating_user_vertices,
                rating_product_vertices,
                data={'inv': torch.zeros(self.ratings.shape[0], dtype=torch.uint8),
                      'rating': torch.FloatTensor(self.ratings['rating'])})
        g.add_edges(
                rating_product_vertices,
                rating_user_vertices,
                data={'inv': torch.ones(self.ratings.shape[0], dtype=torch.uint8),
                      'rating': torch.FloatTensor(self.ratings['rating'])})
        self.g = g

    def refresh_mask(self):
        import torch
        valid_tensor = torch.from_numpy(self.ratings['valid'].values.astype('uint8'))
        test_tensor = torch.from_numpy(self.ratings['test'].values.astype('uint8'))
        train_tensor = torch.from_numpy(self.ratings['train'].values.astype('uint8'))
        prior_tensor = torch.from_numpy(self.ratings['prior'].values.astype('uint8'))
        edge_data = {
                'prior': torch.cat([prior_tensor, prior_tensor], 0),
                'train': torch.cat([train_tensor, train_tensor], 0),
                'test': torch.cat([test_tensor, test_tensor], 0),
                'valid': torch.cat([valid_tensor, valid_tensor], 0)
                }

        self.g.edges[torch.arange(0, len(self.ratings) * 2)].data.update(edge_data)
