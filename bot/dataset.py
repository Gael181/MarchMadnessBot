import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "datasets")
CSV_FILE = os.path.join(DATA_PATH, "DEV _ March Madness.csv")
FAISS_FILE = os.path.join(DATA_PATH, "march_madness_index.faiss")
MODEL_NAME = "all-MiniLM-L6-v2"

df = None
model = None
index = None
faiss_module = None
_initialized = False

def get_team_name(row):
    team_name = row.get("Correct_Team_Name?")

    if pd.isna(team_name) or team_name == "":
        team_name = row.get("Mapped_ESPN_Team_Name")

    if pd.isna(team_name) or team_name == "":
        team_name = row.get("Full_Team_Name")

    if pd.isna(team_name) or team_name == "":
        team_name = "Unknown Team"

    return team_name


def row_to_text(row):
    team_name = get_team_name(row)

    if pd.isna(team_name) or team_name == "":
        team_name = row.get("Mapped_ESPN_Team_Name")

    if pd.isna(team_name) or team_name == "":
        team_name = row.get("Full_Team_Name")

    if pd.isna(team_name) or team_name == "":
        team_name = "Unknown Team"

    return (
        f"{team_name} "
        f"({row.get('Season', 'N/A')} season) "
        f"in the {row.get('Short_Conference_Name', 'Unknown Conference')} conference. "
        f"Offensive Efficiency: {row.get('Adjusted_Offensive_Efficiency', 'N/A')}, "
        f"Defensive Efficiency: {row.get('Adjusted_Defensive_Efficiency', 'N/A')}, "
        f"Tempo: {row.get('Adjusted_Temo', row.get('Adjusted_Tempo', 'N/A'))}, "
        f"Net Rating: {row.get('Net_Rating', 'N/A')}, "
        f"Seed: {row.get('Seed', 'N/A')}, "
        f"Region: {row.get('Region', 'N/A')}."
    )


def initialize():
    global df, model, index, faiss_module, _initialized

    if _initialized:
        return

    import faiss
    from sentence_transformers import SentenceTransformer

    faiss_module = faiss

    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"CSV file not found: {CSV_FILE}")

    df = pd.read_csv(CSV_FILE, low_memory=False)
    df.columns = (
        df.columns
        .str.strip()
        .str.replace('# ', '', regex=False)
        .str.replace(' ', '_', regex=False)
    )

    if "text_chunk" not in df.columns:
        df["text_chunk"] = df.apply(row_to_text, axis=1)
        df.to_csv(CSV_FILE, index=False)

    model = SentenceTransformer(MODEL_NAME)

    if os.path.exists(FAISS_FILE):
        index = faiss_module.read_index(FAISS_FILE)
    else:
        texts = df["text_chunk"].fillna("").tolist()
        embeddings = model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        dimension = embeddings.shape[1]
        index = faiss_module.IndexFlatL2(dimension)
        index.add(embeddings)
        faiss_module.write_index(index, FAISS_FILE)

    _initialized = True


def rebuild_index():
    global index

    initialize()

    texts = df["text_chunk"].fillna("").tolist()
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    dimension = embeddings.shape[1]
    index = faiss_module.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss_module.write_index(index, FAISS_FILE)

    return len(texts)


def search(query, top_k=3):
    initialize()

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding, dtype=np.float32)

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        row = df.iloc[idx]
        results.append({
            "rank": rank + 1,
            "text": row["text_chunk"],
            "team": get_team_name(row),
            "season": row.get("Season", "N/A"),
            "conference": row.get("Short_Conference_Name", "N/A"),
            "seed": row.get("Seed", "N/A"),
            "region": row.get("Region", "N/A"),
            "distance": float(distances[0][rank]),
        })

    return results