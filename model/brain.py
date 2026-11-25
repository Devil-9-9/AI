import torch, torch.nn as nn, torch.nn.functional as F, math
from model.diamond_node import DiamondLattice
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
class DiamondTransformerBrain(nn.Module):
    def __init__(self, vocab_size=50000, d_model=512, nhead=8, num_nodes=12, num_lattices=3):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, 512)
        self.lattices = nn.ModuleList([DiamondLattice(d_model, nhead, num_nodes) for _ in range(num_lattices)])
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
    def causal_mask(self, T):
        return torch.tril(torch.ones(T, T)).view(1, 1, T, T).to(DEVICE)
    def forward(self, idx, max_depth=6):
        B, T = idx.shape
        x = self.token_emb(idx) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        mask = self.causal_mask(T)
        for lattice in self.lattices:
            x = lattice(x, mask, max_depth=max_depth)
        x = self.final_norm(x)
        return self.lm_head(x)
    def generate(self, idx, max_new_tokens=50, temp=0.8, top_k=50):
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -512:]
                logits = self(idx_cond)
                logits = logits[:, -1, :] / temp
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, nxt), dim=1)
        return idx
