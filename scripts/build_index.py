from bot.dataset import rebuild_index

if __name__ == "__main__":
    datasets = ["teams", "tournament"]

    for ds in datasets:
        count = rebuild_index(ds)
        print(f"Built FAISS index for '{ds}' dataset: {count} rows")