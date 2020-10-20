import torch
import torch.utils.data
import tqdm
from sampler import *
from data import *
from model import *
from collections import Counter, defaultdict

# Hyperparameters
BATCH_SIZE = 128                    # batch size
N_SAMPLE_NODES_PER_TYPE = 520       # number of nodes to sample per node type per sampler step
N_SAMPLE_STEPS = 6                  # number of sampler steps
N_WORKERS = 6                       # number of sampler worker processes running in parallel
N_EPOCHS = 10                       # number of epochs to train
N_EVAL_PASSES = 1                   # number of evaluation passes
N_HIDDEN = 512                      # hidden dimension size
N_HEADS = 8                         # number of attention heads
N_LAYERS = 4                        # number of HGT layers
MODEL_PATH = 'model.pkl'            # best model path

hg, g, n_ntypes, n_etypes, n_classes = load_ogb_mag()
sampler = HGTSampler(g, n_ntypes, n_etypes, N_SAMPLE_NODES_PER_TYPE, N_SAMPLE_STEPS)
dl = torch.utils.data.DataLoader(
    g.ndata['train_mask'].nonzero(as_tuple=True)[0],
    batch_size=BATCH_SIZE,
    collate_fn=sampler.sample_subgraph,
    num_workers=N_WORKERS,
    shuffle=True,
    drop_last=False
)
model = HGT(g.ndata['feat'].shape[1], N_HIDDEN, N_HEADS, n_etypes, n_ntypes, N_LAYERS, n_classes)
model = model.cuda()
opt = torch.optim.Adam(model.parameters())

best_val_acc = 0

def train(dl, model):
    model.train()
    with tqdm.tqdm(dl) as tq:
        for i, (sg, num_seed_nodes) in enumerate(tq):
            x = sg.ndata.pop('feat')
            label = sg.ndata.pop('label')
            train_mask = sg.ndata.pop('train_mask')
            val_mask = sg.ndata.pop('val_mask')
            test_mask = sg.ndata.pop('test_mask')

            sg = sg.to('cuda')
            x = x.to('cuda')
            y = model(sg, x)
            label = label.to(y.device)

            y_train = y[train_mask]
            label_train = label[train_mask]
            loss = F.cross_entropy(y_train, label_train)
            opt.zero_grad()
            loss.backward()
            opt.step()

            acc = (y_train.argmax(1) == label_train).float().mean().item()
            loss = loss.item()
            tq.set_postfix({'acc': '%.3f' % acc, 'loss': '%.3f' % loss}, refresh=False)

def evaluate(dl, model):
    # For evaluation we iterate over the data loader several times and take the most common
    # node prediction across their appearances.
    model.eval()
    preds = defaultdict(Counter)
    for _ in range(N_EVAL_PASSES):
        with tqdm.tqdm(dl) as tq, torch.no_grad():
            for i, (sg, num_seed_nodes) in enumerate(tq):
                x = sg.ndata.pop('feat')
                label = sg.ndata.pop('label')

                sg = sg.to('cuda')
                x = x.to('cuda')
                y = model(sg, x).argmax(1).cpu().numpy()
                nid = sg.ndata[dgl.NID].cpu().numpy()

                for y_v, v in zip(y, nid):
                    preds[v].update([y_v])
    final_preds = torch.zeros(g.num_nodes())
    for v, counter_v in preds.items():
        final_preds[v] = counter_v.most_common(1)[0][0]
    all_labels = g.ndata['label']
    all_train_mask = g.ndata['train_mask']
    all_val_mask = g.ndata['val_mask']
    all_test_mask = g.ndata['test_mask']
    train_acc = (final_preds[all_train_mask] == all_labels[all_train_mask]).float().mean().item()
    val_acc = (final_preds[all_val_mask] == all_labels[all_val_mask]).float().mean().item()
    test_acc = (final_preds[all_test_mask] == all_labels[all_test_mask]).float().mean().item()
    print('Train:', train_acc, 'Validation:', val_acc, 'Test:', test_acc)
    return train_acc, val_acc, test_acc

for _ in range(N_EPOCHS):
    train(dl, model)
    _, val_acc, _ = evaluate(dl, model)
    # Save best model
    if best_val_acc < val_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)

