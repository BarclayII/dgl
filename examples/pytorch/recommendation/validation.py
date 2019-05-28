import torch
import tqdm
import numpy as np
from rec.utils import cuda

def compute_validation_ml(ml, h, model, test):
    rr = []
    validation = not test
    n_users = len(ml.users)
    n_products = len(ml.products)

    h = h.cpu()
    M = h[:n_users] @ h[n_users:].t()
    p_nids_candidates = ml.p_nids_candidate_valid if not test else ml.p_nids_candidate_test

    with torch.no_grad():
        with tqdm.trange(n_users) as tq:
            for u_nid in tq:
                p_nids = ml.p_nids[u_nid]
                p_nids_candidate = p_nids_candidates[u_nid]

                score = M[u_nid, p_nids]
                score_sort_idx = score.sort(descending=True)[1].numpy()[:25]

                rank_map = {v: i for i, v in enumerate(p_nids[score_sort_idx])}
                rank_candidates = np.array([rank_map[p_nid] for p_nid in p_nids_candidate if p_nid in rank_map])
                rank = 1 / (rank_candidates + 1)
                rr.append(rank.mean() if len(rank) > 0 else 0.)
                tq.set_postfix({'rank': rank.mean()})

    return np.array(rr)


def compute_validation_cikm(ml, h, model, test):
    rr = []
    outfile = open('submission.txt' if test else 'verify.txt', 'w')

    with torch.no_grad():
        queries = ml.test_queries if test else ml.train_queries_with_clicks
        mrr = 0
        count = 0
        with tqdm.tqdm(queries.iterrows()) as tq:
            for idx, row in tq:
                uid = row['userId']
                items = [ml.product_ids_invmap[int(i)]
                         for i in row['items'].split(',')
                         if int(i) in ml.product_ids_invmap]
                unknown_items = [i for i in row['items'].split(',')
                                 if int(i) not in ml.product_ids_invmap]
                h_dst = h[[i + len(ml.users) for i in items]]

                if np.isnan(uid):
                    h_src = 0
                else:
                    if int(uid) not in ml.user_ids_invmap:
                        #print('%d: %d not showing up in ml.user_ids_invmap' % (row['queryId'], uid))
                        h_src = 0
                    else:
                        uid = ml.user_ids_invmap[int(uid)]
                        h_src = h[uid]
                if len(unknown_items) > 0:
                    #print('%d: unknown items %s' % (row['queryId'], unknown_items))
                    pass

                if isinstance(row['searchstring.tokens'], str):     # i.e. not nan
                    tokens = cuda(torch.LongTensor(
                        [int(i) for i in row['searchstring.tokens'].split(',')]))
                    h_src = h_src + model.emb['tokens'](tokens).mean(0)
                if row['categoryId']:
                    category = cuda(torch.tensor(int(row['categoryId'])))
                    h_src = h_src + model.emb['category'](category)

                score = (h_dst * h_src).sum(1)
                score_sort_idx = score.sort(descending=True)[1].cpu().numpy()
                reordered_items = [ml.product_ids[items[i]] for i in score_sort_idx]
                output = ','.join([str(i) for i in reordered_items] + unknown_items)
                print(row['queryId'], output, file=outfile)

                if not test:
                    rated_products = ml.ratings[ml.ratings['user_id'] == ml.user_ids[uid]]['product_id'].values
                    contained = np.isin(reordered_items, rated_products)
                    if not contained.any():
                        #print('%d: interacted products %s\n %d: queries %s' %
                        #       (row['queryId'], rated_products, row['queryId'], reordered_items))
                        continue
                    rr = np.argmax(contained) + 1
                    mrr = (mrr * idx + 1 / rr) / (idx + 1)
                    tq.set_postfix({'rr': rr, 'mrr': '%.6f' % mrr})

    outfile.close()

    if test:
        import sh
        sh.cp('submission.txt', '/efs/quagan/diginetica/res')
        sh.python('/efs/quagan/diginetica/score.py', '/efs/quagan/diginetica', '/efs/quagan/diginetica/out')
        sh.tee(sh.cat('/efs/quagan/diginetica/out/scores.txt'), 'test.log')
    return np.array([0.])