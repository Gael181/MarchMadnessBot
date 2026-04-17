import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "datasets")

DATASETS = {
    "teams": {
        "csv": "DEV _ March Madness.csv",
        "faiss": "march_madness_index.faiss"
    },
    "tournament": {
        "csv": "ncaa_tournament_results.csv",
        "faiss": "tournament_index.faiss"
    }
}

MODEL_NAME = "all-MiniLM-L6-v2"

STORE = {
    "teams": {"df": None, "index": None},
    "tournament": {"df": None, "index": None},
}

model = None
faiss_module = None
_initialized = False
_current_dataset = None

def is_Upset(row):
    try:
        seed1 = int(row.get("seed"))
        seed2 = int(row.get("seed.1"))
    except:
        return False
    
    score1 = row.get("score")
    score2 = row.get("score.1")

    try:
        if int(score1) > int(score2):
            return seed1 > seed2
        else:
            return seed2 > seed1
    except:
        return False
    

def get_team_name(row):
    team_name = row.get("Correct_Team_Name?")

    if pd.isna(team_name) or team_name == "":
        team_name = row.get("Mapped_ESPN_Team_Name")

    if pd.isna(team_name) or team_name == "":
        team_name = row.get("Full_Team_Name")

    if pd.isna(team_name) or team_name == "":
        team_name = "Unknown Team"

    return team_name


def row_to_text_teams(row):
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

def row_to_text_tournament(row):
    team1 = row.get("team", "Unknown Team")
    team2 = row.get("team.1", "Unknown Team")

    seed1 = row.get("seed", "N/A")
    seed2 = row.get("seed.1", "N/A")

    score1 = row.get("score", "N/A")
    score2 = row.get("score.1", "N/A")

    upset = "UPSET" if is_Upset(row) else "no upset"
    
    return (
        f"NCAA Tournament Game ({row.get('year', 'N/A')}), "
        f"Round {row.get('round', 'N/A')}, "
        f"{row.get('region_name', 'Unknown Region')} region. "
        f"{team1} (Seed {seed1}) {score1} - "
        f"{team2} (Seed {seed2}) {score2}. "
        f"... Upset: {upset}."
    )

def _get_row_text(dataset_name, row):
    if dataset_name == "tournament":
        return row_to_text_tournament(row)
    return row_to_text_teams(row)

def initialize(dataset_name="teams"):
    global model, faiss_module, _initialized, _current_dataset

    if _initialized and _current_dataset == dataset_name:
        return

    import faiss
    from sentence_transformers import SentenceTransformer

    faiss_module = faiss

    if model is None:
        model = SentenceTransformer(MODEL_NAME)

    config = DATASETS[dataset_name]

    csv_file = os.path.join(DATA_PATH, config["csv"])
    faiss_file = os.path.join(DATA_PATH, config["faiss"])

    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    df = pd.read_csv(csv_file, low_memory=False)
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace('# ', '', regex=False)
        .str.replace(' ', '_', regex=False)
    )
    
    if "text_chunk" not in df.columns:
        df["text_chunk"] = df.apply(lambda r: _get_row_text(dataset_name, r), axis=1)
        df.to_csv(csv_file, index=False)

    if os.path.exists(faiss_file):
        index = faiss_module.read_index(faiss_file)
    else:
        texts = df["text_chunk"].fillna("").tolist()
        embeddings = model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        index = faiss_module.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        faiss_module.write_index(index, faiss_file)

    STORE[dataset_name]["df"] = df
    STORE[dataset_name]["index"] = index
    _current_dataset = dataset_name
    _initialized = True


def rebuild_index(dataset_name="teams"):
    initialize(dataset_name)

    df = STORE[dataset_name]["df"]
    index = STORE[dataset_name]["index"]

    texts = df["text_chunk"].fillna("").tolist()
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    index = faiss_module.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss_file = os.path.join(DATA_PATH, DATASETS[dataset_name]["faiss"])
    faiss_module.write_index(index, faiss_file)

    STORE[dataset_name]["index"] = index

    return len(texts)


def search(query, top_k=3, dataset="teams"):
    initialize(dataset)

    df = STORE[dataset]["df"]
    index = STORE[dataset]["index"]

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