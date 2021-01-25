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
N_WORKERS = 16                      # number of sampler worker processes running in parallel
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
    drop_last=False,
)
model = HGT(g.ndata['feat'].shape[1], N_HIDDEN, N_HEADS, n_etypes, n_ntypes, N_LAYERS, n_classes)
model = model.cuda()
param_optimizer = list(model.named_parameters())
def no_decay(param_name):
    return 'norm' in param_name or param_name.endswith('.bias') or param_name.endswith('.b')
optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not no_decay(n)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if no_decay(n)],     'weight_decay': 0.0}]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=1e-06)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, pct_start=0.05, anneal_strategy='linear', final_div_factor=10,\
                        max_lr = 5e-4, total_steps = len(dl) * N_EPOCHS + 1)

best_val_acc = 0
train_step = 0
exceptional = 0
log = open('train.log', 'w')

def train(dl, model):
    global train_step
    model.train()
    with tqdm.tqdm(dl) as tq:
        for i, (sg, num_seed_nodes) in enumerate(tq):
            if sg.num_edges() > 200000:
                with open('test%d.pkl' % exceptional, 'wb') as f:
                    pickle.dump(sg, f)
                    exceptional += 1
                    continue

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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_step += 1
            scheduler.step(train_step)

            acc = (y_train.argmax(1) == label_train).float().mean().item()
            val_acc = (y[val_mask].argmax(1) == label[val_mask]).float().mean().item()
            test_acc = (y[test_mask].argmax(1) == label[test_mask]).float().mean().item()
            loss = loss.item()
            tq.set_postfix({
                'acc': '%.3f' % acc, 'loss': '%.3f' % loss,
                'val_acc': '%.3f' % val_acc, 'test_acc': '%.3f' % test_acc}, refresh=False)
            if i % 100 == 99:
                print(
                    'acc', '%.3f' % acc, 'loss', '%.3f' % loss,
                    'val_acc', '%.3f' % val_acc, 'test_acc', '%.3f' % test_acc, file=log, flush=True)

def evaluate(dl, model):
    # For evaluation we iterate over the data loader several times and take the most common
    # node prediction across their appearances.
    model.eval()

    sum_preds = torch.zeros(g.num_nodes(), n_classes).cuda()
    counts = torch.zeros(g.num_nodes()).cuda()
    for _ in range(N_EVAL_PASSES):
        with tqdm.tqdm(dl) as tq, torch.no_grad():
            for i, (sg, num_seed_nodes) in enumerate(tq):
                x = sg.ndata.pop('feat')
                label = sg.ndata.pop('label')

                sg = sg.to('cuda')
                x = x.to('cuda')
                y = model(sg, x)
                nid = sg.ndata[dgl.NID]

                ones = torch.ones(nid.shape[0]).to(y.device)
                sum_preds.scatter_add_(0, nid[:, None].expand_as(y), y)
                counts.scatter_add_(0, nid, ones)
    avg_preds = sum_preds / counts[:, None]
    final_preds = avg_preds.argmax(1)

    all_labels = g.ndata['label'].cuda()
    all_train_mask = g.ndata['train_mask'].cuda()
    all_val_mask = g.ndata['val_mask'].cuda()
    all_test_mask = g.ndata['test_mask'].cuda()
    train_acc = (final_preds[all_train_mask] == all_labels[all_train_mask]).float().mean().item()
    val_acc = (final_preds[all_val_mask] == all_labels[all_val_mask]).float().mean().item()
    test_acc = (final_preds[all_test_mask] == all_labels[all_test_mask]).float().mean().item()
    print('Train:', train_acc, 'Validation:', val_acc, 'Test:', test_acc)
    print('Train:', train_acc, 'Validation:', val_acc, 'Test:', test_acc, file=log, flush=True)
    return train_acc, val_acc, test_acc

for _ in range(N_EPOCHS):
    train(dl, model)
    _, val_acc, _ = evaluate(dl, model)
    # Save best model
    if best_val_acc < val_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
log.close()
