import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, random

st.title("💎 Diamond Brain - Chat Karo")

# ===== SAB KUCH ISI FILE ME HAI, BAHAR SE KUCH NAHI LA RAHE =====

class DiamondTokenizer:
    def encode(self, s):
        return [ord(c) % 50000 for c in s]
    def decode(self, ids):
        return ''.join([chr(i % 256) for i in ids])

class DiamondAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.nhead = 8
        self.d_k = 512 // 8
        self.q = nn.Linear(512, 512); self.k = nn.Linear(512, 512); self.v = nn.Linear(512, 512)
        self.out = nn.Linear(512, 512)
        self.norm = nn.LayerNorm(512)
    def forward(self, x):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, 8, self.d_k).transpose(1, 2)
        k = self.k(x).view(B, T, 8, self.d_k).transpose(1, 2)
        v = self.v(x).view(B, T, 8, self.d_k).transpose(1, 2)
        s = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T)
        s = s.masked_fill(mask == 0, -1e9)
        w = F.softmax(s, dim=-1)
        o = torch.matmul(w, v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.norm(x + self.out(o))

class DiamondBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(50000, 512)
        self.attn1 = DiamondAttention()
        self.attn2 = DiamondAttention()
        self.ffn = nn.Sequential(nn.Linear(512, 2048), nn.GELU(), nn.Linear(2048, 512))
        self.norm = nn.LayerNorm(512)
        self.head = nn.Linear(512, 50000)
    def forward(self, idx):
        x = self.emb(idx)
        x = self.attn1(x)
        x = self.attn2(x)
        x = self.norm(x + self.ffn(x))
        return self.head(x)
    def generate(self, idx, max_new_tokens=30):
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -128:]
                logits = self(idx_cond)
                logits = logits[:, -1, :] / 0.8
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, nxt), dim=1)
        return idx

@st.cache_resource
def load_model():
    model = DiamondBrain()
    try:
        model.load_state_dict(torch.load("diamond_brain_500.pt", map_location="cpu"))
        st.success("Trained Dimag Load Ho Gaya!")
    except:
        st.warning("Abhi Training nahi hui, naya dimag use ho raha hai")
    model.eval()
    return model, DiamondTokenizer()

model, tok = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Diamond se kuch pucho..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Diamond soch raha hai..."):
            ids = torch.tensor([tok.encode(prompt)])
            out = model.generate(ids, max_new_tokens=25)
            reply = tok.decode(out[0].tolist())
            st.write(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
