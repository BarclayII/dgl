import torch
import tqdm
import numpy as np
from rec.utils import cuda

def compute_validation_ml(ml, h, model):
    rr = []

    with torch.no_grad():
        with tqdm.trange(n_users) as tq:
            for u_nid in tq:
                uid = ml.user_ids[u_nid]
                pids_exclude = ml.ratings[
                        (ml.ratings['user_id'] == uid) &
                        (ml.ratings['train'] | ml.ratings['test' if validation else 'valid'])
                        ]['product_id'].values
                pids_candidate = ml.ratings[
                        (ml.ratings['user_id'] == uid) &
                        ml.ratings['valid' if validation else 'test']
                        ]['product_id'].values
                pids = np.setdiff1d(ml.product_ids, pids_exclude)
                p_nids = np.array([ml.product_ids_invmap[pid] for pid in pids])
                p_nids_candidate = np.array([ml.product_ids_invmap[pid] for pid in pids_candidate])

                dst = torch.from_numpy(p_nids) + n_users
                src = torch.zeros_like(dst).fill_(u_nid)
                h_dst = h[dst]
                h_src = h[src]

                score = (h_src * h_dst).sum(1)
                score_sort_idx = score.sort(descending=True)[1].cpu().numpy()

                rank_map = {v: i for i, v in enumerate(p_nids[score_sort_idx])}
                rank_candidates = np.array([rank_map[p_nid] for p_nid in p_nids_candidate])
                rank = 1 / (rank_candidates + 1)
                rr.append(rank.mean())
                tq.set_postfix({'rank': rank.mean()})

    return np.array(rr)


def compute_validation_cikm(ml, h, model, test):
    rr = []
    outfile = open('submission.txt' if test else 'verify.txt', 'w')

    with torch.no_grad():
        queries = ml.test_queries if test else ml.train_queries_with_clicks
        with tqdm.tqdm(queries.iterrows()) as tq:
            for idx, row in tq:
                uid = row['userId']
                items = [ml.product_ids_invmap[int(i)]
                         for i in row['items'].split(',')
                         if int(i) in ml.product_ids_invmap]
                unknown_items = [i for i in row['items'].split(',')
                                 if int(i) not in ml.product_ids_invmap]
                h_dst = h[items]

                if np.isnan(uid):
                    h_src = 0
                else:
                    if int(uid) not in ml.user_ids_invmap:
                        print('%d: %d not showing up in ml.user_ids_invmap' % (row['queryId'], uid))
                        h_src = 0
                    else:
                        uid = ml.user_ids_invmap[int(uid)]
                        h_src = h[uid]
                if len(unknown_items) > 0:
                    print('%d: unknown items %s' % (row['queryId'], unknown_items))

                if isinstance(row['searchstring.tokens'], str):     # i.e. not nan
                    tokens = cuda(torch.LongTensor(
                        [int(i) for i in row['searchstring.tokens'].split(',')]))
                    h_src += model.emb['tokens'](tokens).mean(0)
                if row['categoryId']:
                    category = cuda(torch.tensor(int(row['categoryId'])))
                    h_src += model.emb['category'](category)

                score = (h_dst * h_src).sum(1)
                score_sort_idx = score.sort(descending=True)[1].cpu().numpy()
                output = ','.join(
                        [str(ml.product_ids[items[i]]) for i in score_sort_idx] +
                        unknown_items)
                print(row['queryId'], output, file=outfile)
    outfile.close()
    return np.array([0.])
