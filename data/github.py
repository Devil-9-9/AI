from datasets import load_dataset
def get_github_data(limit=2000):
    print("Loading GitHub Code...")
    texts = []
    try:
        ds = load_dataset("bigcode/starcoderdata", split="train", streaming=True)
        for i, ex in enumerate(ds):
            if ex.get("content"):
                texts.append(ex["content"][:2000])
            if i >= limit:
                break
    except Exception as e:
        print("GitHub fail:", e)
    print(f"GitHub loaded: {len(texts)}")
    return texts
