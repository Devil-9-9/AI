from datasets import load_dataset
def get_common_crawl_data(limit=2000):
    print("Loading Common Crawl / OpenWebText...")
    texts = []
    try:
        ds = load_dataset("openwebtext", split="train", streaming=True)
        for i, ex in enumerate(ds):
            if ex.get("text"):
                texts.append(ex["text"][:2000])
            if i >= limit:
                break
    except Exception as e:
        print("CommonCrawl fail:", e)
    print(f"CommonCrawl loaded: {len(texts)}")
    return texts
