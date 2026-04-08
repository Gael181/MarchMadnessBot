from bot.dataset import rebuild_index

if __name__ == "__main__":
    count = rebuild_index()
    print(f"Built FAISS index for {count} rows.")