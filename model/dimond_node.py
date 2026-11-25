import torch, torch.nn as nn, random
from model.attention import DiamondAttention, DiamondFeedForward
class DiamondCarbonNode(nn.Module):
    def __init__(self, d_model=512, nhead=8, node_id=0):
        super().__init__()
        self.node_id = node_id
        self.attn = DiamondAttention(d_model, nhead)
        self.ffn = DiamondFeedForward(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.bonds = nn.ModuleList()
        self.alive = True
        self.memory = None
    def add_bond(self, node):
        if node not in self.bonds:
            self.bonds.append(node)
    def kill(self): self.alive = False
    def revive(self): self.alive = True
    def forward(self, x, mask=None, depth=0, max_depth=6, visited=None):
        if not self.alive: return x
        if visited is None: visited = set()
        if self.node_id in visited and depth > 2: return x
        visited = visited | {self.node_id}
        if depth >= max_depth: return x
        try:
            h = self.attn(x, mask)
            h = self.ffn(h)
            if self.memory is not None and self.memory.shape == h.shape:
                h = h * 0.9 + self.memory * 0.1
            self.memory = h.detach().clone()
            if len(self.bonds) == 0: return h
            outs = []
            for bond in self.bonds:
                if bond.alive:
                    try: outs.append(bond(h, mask, depth+1, max_depth, visited))
                    except Exception: continue
            if len(outs) == 0: return h
            stacked = torch.stack(outs, dim=0).mean(dim=0)
            return self.norm2(h + stacked) / 2
        except Exception:
            return x
class DiamondLattice(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_nodes=12):
        super().__init__()
        self.nodes = nn.ModuleList([DiamondCarbonNode(d_model, nhead, i) for i in range(num_nodes)])
        for i in range(num_nodes):
            for j in range(1, 5):
                self.nodes[i].add_bond(self.nodes[(i+j) % num_nodes])
            self.nodes[i].add_bond(self.nodes[(i-1) % num_nodes])
    def forward(self, x, mask=None, max_depth=6):
        return self.nodes[0](x, mask, 0, max_depth, set())
