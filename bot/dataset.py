import re
import uuid
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
    },
    "admin_uploads": {
        "csv": "admin_uploads.csv",
        "faiss": "admin_uploads_index.faiss"
    }
}

MODEL_NAME = "all-MiniLM-L6-v2"

STORE = {
    "teams": {"df": None, "index": None},
    "tournament": {"df": None, "index": None},
    "admin_uploads": {"df": None, "index": None},
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
    team_name = row.get("correct_team_name?")

    if pd.isna(team_name) or team_name == "":
        team_name = row.get("mapped_espn_team_name")

    if pd.isna(team_name) or team_name == "":
        team_name = row.get("full_team_name")

    if pd.isna(team_name) or team_name == "":
        team_name = "Unknown Team"

    return team_name


def row_to_text_teams(row):
    team_name = get_team_name(row)

    if pd.isna(team_name) or team_name == "":
        team_name = row.get("mapped_espn_team_name")

    if pd.isna(team_name) or team_name == "":
        team_name = row.get("full_team_name")

    if pd.isna(team_name) or team_name == "":
        team_name = "Unknown Team"

    return (
        f"{team_name} "
        f"({row.get('season', 'N/A')} season) "
        f"in the {row.get('short_conference_name', 'unknown conference')} conference. "
        f"Offensive Efficiency: {row.get('adjusted_offensive_efficiency', 'N/A')}, "
        f"Defensive Efficiency: {row.get('adjusted_defensive_efficiency', 'N/A')}, "
        f"Tempo: {row.get('adjusted_tempo', row.get('adjusted_tempo', 'N/A'))}, "
        f"Net Rating: {row.get('net_rating', 'N/A')}, "
        f"Seed: {row.get('seed', 'N/A')}, "
        f"Region: {row.get('region', 'N/A')}."
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
    
def clean_text(value):
    if pd.isna(value):
        return ""

    value = str(value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def row_to_text_admin_uploads(row):
    title = row.get("title", "")
    source = row.get("source", "admin upload")
    content = row.get("content", "")

    return clean_text(f"Source: {source}. Title: {title}. Content: {content}")


def clean_dataframe(df, dataset_name="teams"):
    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace("# ", "", regex=False)
        .str.replace(" ", "_", regex=False)
    )

    df = df.rename(columns={
        "adjusted_temo": "adjusted_tempo"
    })

    df = df.dropna(how="all")
    df = df.drop_duplicates()

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(clean_text)

    if dataset_name == "admin_uploads":
        if "content" not in df.columns:
            text_columns = [col for col in df.columns if col != "text_chunk"]
            df["content"] = df[text_columns].astype(str).agg(" ".join, axis=1)

        if "title" not in df.columns:
            df["title"] = "Admin uploaded document"

        if "source" not in df.columns:
            df["source"] = "admin upload"

    if "text_chunk" not in df.columns:
        df["text_chunk"] = df.apply(lambda r: _get_row_text(dataset_name, r), axis=1)

    df["text_chunk"] = df["text_chunk"].apply(clean_text)
    df = df[df["text_chunk"] != ""]

    return df   

def _get_row_text(dataset_name, row):
    if dataset_name == "tournament":
        return row_to_text_tournament(row)

    if dataset_name == "admin_uploads":
        return row_to_text_admin_uploads(row)

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
    df = clean_dataframe(df, dataset_name)
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

def _reset_dataset_cache(dataset_name):
    global _initialized, _current_dataset

    STORE[dataset_name]["df"] = None
    STORE[dataset_name]["index"] = None
    _initialized = False
    _current_dataset = None


def _save_and_rebuild(df, dataset_name="admin_uploads"):
    os.makedirs(DATA_PATH, exist_ok=True)

    csv_file = os.path.join(DATA_PATH, DATASETS[dataset_name]["csv"])

    if os.path.exists(csv_file):
        old_df = pd.read_csv(csv_file, low_memory=False)
        old_df = clean_dataframe(old_df, dataset_name)
        df = pd.concat([old_df, df], ignore_index=True)

    df = clean_dataframe(df, dataset_name)
    df = df.drop_duplicates(subset=["text_chunk"])

    df.to_csv(csv_file, index=False)

    _reset_dataset_cache(dataset_name)
    return rebuild_index(dataset_name)


def ingest_uploaded_csv(uploaded_file, dataset_name="admin_uploads"):
    if uploaded_file is None:
        raise ValueError("No CSV file uploaded.")

    df = pd.read_csv(uploaded_file, low_memory=False)
    df = clean_dataframe(df, dataset_name)

    return _save_and_rebuild(df, dataset_name)


def ingest_raw_text(raw_text, title="Admin pasted text", dataset_name="admin_uploads"):
    raw_text = clean_text(raw_text)

    if not raw_text:
        raise ValueError("No text was provided.")

    df = pd.DataFrame([
        {
            "id": str(uuid.uuid4()),
            "title": title or "Admin pasted text",
            "source": "manual text upload",
            "content": raw_text,
        }
    ])

    df = clean_dataframe(df, dataset_name)

    return _save_and_rebuild(df, dataset_name)


def search(query, top_k=3, dataset="teams"):
    initialize(dataset)

    df = STORE[dataset]["df"]
    index = STORE[dataset]["index"]
    if df is None or df.empty or index is None or index.ntotal == 0:
        return []
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
            "season": row.get("season", "N/A"),
            "conference": row.get("short_conference_name", "N/A"),
            "seed": row.get("seed", "N/A"),
            "region": row.get("region", "N/A"),
            "distance": float(distances[0][rank]),
        })

    return results

def get_dataset_summary():
    summaries = []

    for dataset_name, config in DATASETS.items():
        csv_file = os.path.join(DATA_PATH, config["csv"])
        faiss_file = os.path.join(DATA_PATH, config["faiss"])

        row_count = 0
        exists = os.path.exists(csv_file)

        if exists:
            try:
                df = pd.read_csv(csv_file, low_memory=False)
                row_count = len(df)
            except Exception:
                row_count = 0

        summaries.append({
            "name": dataset_name,
            "csv": config["csv"],
            "faiss": config["faiss"],
            "csv_exists": exists,
            "faiss_exists": os.path.exists(faiss_file),
            "row_count": row_count,
            "can_delete": dataset_name == "admin_uploads",
        })

    return summaries


def list_admin_upload_rows():
    csv_file = os.path.join(DATA_PATH, DATASETS["admin_uploads"]["csv"])

    if not os.path.exists(csv_file):
        return []

    df = pd.read_csv(csv_file, low_memory=False)

    if df.empty:
        return []

    rows = []

    for index, row in df.iterrows():
        title = row.get("title", "Uploaded data")
        source = row.get("source", "admin upload")
        content = row.get("content", "")
        text_chunk = row.get("text_chunk", "")

        preview = content if str(content).strip() else text_chunk
        preview = str(preview).replace("\n", " ").strip()

        if len(preview) > 20:
            preview = preview[:25] + "..."

        rows.append({
            "index": index,
            "title": title if str(title).strip() else "Uploaded data",
            "source": source if str(source).strip() else "admin upload",
            "preview": preview,
        })

    return rows


def _remove_file_if_exists(path):
    if os.path.exists(path):
        os.remove(path)


def delete_admin_upload_row(row_index):
    csv_file = os.path.join(DATA_PATH, DATASETS["admin_uploads"]["csv"])
    faiss_file = os.path.join(DATA_PATH, DATASETS["admin_uploads"]["faiss"])

    if not os.path.exists(csv_file):
        raise FileNotFoundError("No admin upload dataset exists yet.")

    df = pd.read_csv(csv_file, low_memory=False)

    if row_index < 0 or row_index >= len(df):
        raise IndexError("Selected row does not exist.")

    df = df.drop(index=row_index).reset_index(drop=True)

    if df.empty:
        df.to_csv(csv_file, index=False)
        _remove_file_if_exists(faiss_file)
        _reset_dataset_cache("admin_uploads")
        return 0

    df = clean_dataframe(df, "admin_uploads")
    df.to_csv(csv_file, index=False)

    _reset_dataset_cache("admin_uploads")
    return rebuild_index("admin_uploads")


def clear_admin_uploads():
    csv_file = os.path.join(DATA_PATH, DATASETS["admin_uploads"]["csv"])
    faiss_file = os.path.join(DATA_PATH, DATASETS["admin_uploads"]["faiss"])

    _remove_file_if_exists(csv_file)
    _remove_file_if_exists(faiss_file)
    _reset_dataset_cache("admin_uploads")

    return 0