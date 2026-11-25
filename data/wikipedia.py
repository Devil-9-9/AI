from datasets import load_dataset
def get_wikipedia_books_data(limit=2000):
    print("Loading Wikipedia + Books...")
    texts = []
    try:
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
        for i, ex in enumerate(ds):
            if ex.get("text") and len(ex["text"]) > 50:
                texts.append(ex["text"][:2000])
            if i >= limit:
                break
    except Exception as e:
        print("Wiki fail:", e)
    try:
        ds2 = load_dataset("bookcorpus", split="train", streaming=True)
        for i, ex in enumerate(ds2):
            if ex.get("text"):
                texts.append(ex["text"][:2000])
            if i >= 1000:
                break
    except Exception as e:
        print("Book fail:", e)
    print(f"Wiki+Books loaded: {len(texts)}")
    return texts
