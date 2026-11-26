import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from data.common_crawl import get_common_crawl_data
from data.wikipedia import get_wikipedia_books_data
from data.github import get_github_data
from data.social import get_social_data
from model.tokenizer import DiamondTokenizer
from model.brain import DiamondTransformerBrain, DEVICE
from training.trainer import InfiniteDataset, train_one_step
def infinite_train_loop():
    tok = DiamondTokenizer()
    model = DiamondTransformerBrain().to(DEVICE)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10000)
    texts = []
    texts += get_common_crawl_data()
    texts += get_wikipedia_books_data()
    texts += get_github_data()
    texts += get_social_data()
    print(f"TOTAL 4 SOURCES: {len(texts)} docs")
    dataset = InfiniteDataset(texts, tok, seq_len=128)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    step = 0; cycle = 0
    while True:
        cycle += 1
        print(f"\n=== CYCLE {cycle} INFINITE ===")
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            loss = train_one_step(model, opt, bx, by)
            sched.step(); step += 1
            if step % 10 == 0:
                print(f"Step {step} Loss {loss:.4f}")
            if step % 500 == 0:
                torch.save(model.state_dict(), f"diamond_brain_{step}.pt")
                model.eval()
                start = torch.tensor([tok.encode("Hello Diamond")]).to(DEVICE)
                out = model.generate(start, max_new_tokens=20)
                print("Sample:", tok.decode(out[0].tolist()))
            if step % 2000 == 0:
                break
