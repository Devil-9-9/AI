import torch, torch.nn.functional as F
from torch.utils.data import Dataset
import random
class InfiniteDataset(Dataset):
    def __init__(self, texts, tok, seq_len=128):
        self.tok = tok; self.seq_len = seq_len; self.encoded = []
        for t in texts:
            ids = tok.encode(t)
            if len(ids) > 10:
                self.encoded.append(ids)
        print(f"Encoded {len(self.encoded)} docs")
    def __len__(self): return len(self.encoded) * 1000000
    def __getitem__(self, idx):
        real = idx % len(self.encoded)
        ids = self.encoded[real]
        if len(ids) <= self.seq_len + 1:
            ids = (ids * 10)[:self.seq_len+1]
        else:
            s = random.randint(0, len(ids)-self.seq_len-1)
            ids = ids[s:s+self.seq_len+1]
        return torch.tensor(ids[:-1]), torch.tensor(ids[1:])
def train_one_step(model, opt, bx, by):
    model.train()
    opt.zero_grad()
    logits = model(bx)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), by.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.item()
