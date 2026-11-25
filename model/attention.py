import torch, torch.nn as nn, torch.nn.functional as F, math
class DiamondAttention(nn.Module):
    def __init__(self, d_model=512, nhead=8):
        super().__init__()
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.q1 = nn.Linear(d_model, d_model); self.k1 = nn.Linear(d_model, d_model); self.v1 = nn.Linear(d_model, d_model)
        self.q2 = nn.Linear(d_model, d_model); self.k2 = nn.Linear(d_model, d_model); self.v2 = nn.Linear(d_model, d_model)
        self.q3 = nn.Linear(d_model, d_model); self.k3 = nn.Linear(d_model, d_model); self.v3 = nn.Linear(d_model, d_model)
        self.q4 = nn.Linear(d_model, d_model); self.k4 = nn.Linear(d_model, d_model); self.v4 = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
    def _attn(self, q, k, v, mask=None):
        B, T, _ = q.shape
        q = q.view(B, T, self.nhead, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.nhead, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.nhead, self.d_k).transpose(1, 2)
        s = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            s = s.masked_fill(mask == 0, -1e9)
        w = F.softmax(s, dim=-1)
        o = torch.matmul(w, v).transpose(1, 2).contiguous().view(B, T, -1)
        return o
    def forward(self, x, mask=None):
        o1 = self._attn(self.q1(x), self.k1(x), self.v1(x), mask)
        o2 = self._attn(self.q2(x), self.k2(x), self.v2(x), mask)
        o3 = self._attn(self.q3(x), self.k3(x), self.v3(x), mask)
        o4 = self._attn(self.q4(x), self.k4(x), self.v4(x), mask)
        return self.norm(x + self.out((o1+o2+o3+o4)/4.0))
class DiamondFeedForward(nn.Module):
    def __init__(self, d_model=512, dim_ff=2048):
        super().__init__()
        self.l1 = nn.Linear(d_model, dim_ff); self.l2 = nn.Linear(dim_ff, d_model)
        self.l3 = nn.Linear(d_model, dim_ff); self.l4 = nn.Linear(dim_ff, d_model)
        self.norm = nn.LayerNorm(d_model); self.drop = nn.Dropout(0.1)
    def forward(self, x):
        a = self.l2(F.gelu(self.l1(x)))
        b = self.l4(F.gelu(self.l3(x)))
        return self.norm(x + self.drop(a+b)/2)
