import os
import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "datasets")
CSV_FILE = os.path.join(DATA_PATH, "DEV _ March Madness.csv")
FAISS_FILE = os.path.join(DATA_PATH, "march_madness_index.faiss")


if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE, low_memory=False)
    df.columns = df.columns.str.strip().str.replace('# ', '').str.replace(' ', '_')
else:
    raise FileNotFoundError(f"CSV file not found: {CSV_FILE}")

print("Dataset loaded.")
# print("Sample columns:", df.columns[:10])
# test
# print(df[['Mapped_ESPN_Team_Name', 'Full_Team_Name']].head(10))


print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')


def row_to_text(row):
    team_name = row.get('Correct_Team_Name?')

    if pd.isna(team_name) or team_name == "":
        team_name = row.get('Mapped_ESPN_Team_Name')

    if pd.isna(team_name) or team_name == "":
        team_name = row.get('Full_Team_Name')

    if pd.isna(team_name) or team_name == "":
        team_name = "Unknown Team"

    return (
        f"{team_name} "
        f"({row.get('Season', 'N/A')} season) "
        f"in the {row.get('Short_Conference_Name', 'Unknown Conference')} conference. "
        f"Offensive Efficiency: {row.get('Adjusted_Offensive_Efficiency', 'N/A')}, "
        f"Defensive Efficiency: {row.get('Adjusted_Defensive_Efficiency', 'N/A')}, "
        f"Tempo: {row.get('Adjusted_Temo', 'N/A')}, "
        f"Net Rating: {row.get('Net_Rating', 'N/A')}, "
        f"Seed: {row.get('Seed', 'N/A')}, "
        f"Region: {row.get('Region', 'N/A')}."
    )


print("Rebuilding text chunks...")
df['text_chunk'] = df.apply(row_to_text, axis=1)
df.to_csv(CSV_FILE, index=False)


if os.path.exists(FAISS_FILE):
    print("Loading FAISS index...")
    index = faiss.read_index(FAISS_FILE)
else:
    print("Creating FAISS index...")
    texts = df['text_chunk'].tolist()

    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)

    embeddings = np.array(embeddings, dtype=np.float32)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, FAISS_FILE)
    print("FAISS index saved!")


def search(query, top_k=3):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding, dtype=np.float32)

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        results.append(df.iloc[idx]['text_chunk'])

    return results


def add_row(row_dict):
    global df, index

    row_dict['text_chunk'] = row_to_text(row_dict)

    df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)

    embedding = model.encode([row_dict['text_chunk']])
    embedding = np.array(embedding, dtype=np.float32)

    index.add(embedding)
    faiss.write_index(index, FAISS_FILE)

    print("Row added successfully.")

# test
# if __name__ == "__main__":
#     print("\nTesting retrieval...\n")

#     results = search("best teams in 2026", top_k=3)

#     for i, r in enumerate(results):
#         print(f"Result {i+1}:")
#         print(r)
#         print()