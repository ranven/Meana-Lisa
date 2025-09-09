
    # pip install pymongo requests

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, Any
import time


import requests
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError

MONGODB_URI = "mongodb+srv://admin:2I5At8MJUVPNJgmF@meana-lisa.xswknhv.mongodb.net/"
DB_NAME = "Paintings"
COLL_NAME = "Raw"

# Adjust to your API host. Assumes GET /search?id=<object_id> returns JSON.
BASE_URL = "https://collectionapi.metmuseum.org/public/collection/v1"
SEARCH_PATH = "objects"
TIMEOUT = 15

BIG_DELAY = 12
DELAY = 1

START = 1000 # -> if somehow everything oblitarates, update this number to the last index of list on painting ids


IDS_PATH = Path("data/paintings_id.json")  # JSON array of IDs

def load_ids(path):
    with path.open() as f:
        data = json.load(f)
        
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array of IDs")
    return data

def fetch_search(session: requests.Session, object_id: Any) -> dict:
    r = session.get(f"{BASE_URL}/{SEARCH_PATH}/{object_id}", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def batched(iterable: Iterable, n: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch

def main():
    object_ids = load_ids(IDS_PATH)
    object_ids = object_ids[START:]

    client = MongoClient(MONGODB_URI)
    col = client[DB_NAME][COLL_NAME]
    session = requests.Session()
    
    i = 0
    n = len(object_ids)
    processed = 0

    while i < n:
        oid = object_ids[i]

        if DELAY:
            time.sleep(DELAY)

        try:
            payload = fetch_search(session, oid)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                print(f"[skip-404] {oid}")
                i += 1            # advance past this oid
                continue
            print(f"[retry] {oid}: HTTP {getattr(e.response,'status_code',None)}. sleeping {BIG_DELAY}s")
            time.sleep(BIG_DELAY) # stay on same oid
            continue
        except requests.RequestException as e:
            print(f"[retry] {oid}: http error: {e}. sleeping {BIG_DELAY}s")
            time.sleep(BIG_DELAY) # stay on same oid
            continue

        try:
            col.update_one(
                {"_id": oid},
                {"$set": {
                    "object_id": oid,
                    "data": payload,
                    "fetched_at": datetime.utcnow(),
                }},
                upsert=True,
            )
        except Exception as e:
            print(f"[retry] mongo error for {oid}: {e}. sleeping {BIG_DELAY}s")
            time.sleep(BIG_DELAY) # stay on same oid
            continue

        processed += 1
        print(f"[ok] {START + processed} upserted {oid}")
        i += 1  # advance only after success or a 404 skip

if __name__ == "__main__":
    main()