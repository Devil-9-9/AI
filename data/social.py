from datasets import load_dataset
def get_social_data(limit=2000):
    print("Loading Social / Conversations...")
    texts = []
    try:
        ds = load_dataset("Anthropic/hh-rlhf", split="train", streaming=True)
        for i, ex in enumerate(ds):
            if ex.get("chosen"):
                texts.append(ex["chosen"][:2000])
            if i >= limit:
                break
    except Exception as e:
        print("Social fail:", e)
    print(f"Social loaded: {len(texts)}")
    return texts
