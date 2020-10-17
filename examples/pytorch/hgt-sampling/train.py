import torch
import torch.utils.data
import tqdm
from sampler import *
from data import *
from model import *

g, n_ntypes, n_etypes, n_classes = load_ogb_mag()
STEP = 128
P = 520
L = 6
NWORKERS = 16
NEPOCHS = 2
sampler = HGTSampler(g, n_ntypes, n_etypes, P, L)
dl = torch.utils.data.DataLoader(
    g.ndata['train_mask'].nonzero(as_tuple=True)[0],
    batch_size=STEP,
    collate_fn=sampler.sample_subgraph,
    num_workers=NWORKERS,
    shuffle=True,
    drop_last=False
)
model = HGT(g.ndata['feat'].shape[1], 512, 8, n_etypes, n_ntypes, 4, n_classes)
opt = torch.optim.Adam(model.parameters())

for _ in range(NEPOCHS):
    for sg, num_seed_nodes in tqdm.tqdm(dl):
        x = sg.ndata.pop('feat')
        label = sg.ndata.pop('label')
        train_mask = sg.ndata.pop('train_mask')
        val_mask = sg.ndata.pop('val_mask')
        test_mask = sg.ndata.pop('test_mask')

        y = model(sg, x)
        loss = F.cross_entropy(y[train_mask], label[train_mask].to(y.device))
        opt.zero_grad()
        loss.backward()
        opt.step()
