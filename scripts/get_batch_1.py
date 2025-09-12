
    # pip install pymongo requests

import json
import logging
import sys
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

START = 1028 # -> if somehow everything oblitarates, update this number to the last index of list on painting ids

IDS_PATH = Path("data/paintings_id.json")  # JSON array of IDs
LOGS_DIR = Path("logs")

def setup_logging():
    """Setup logging with timestamped file and console output."""
    # Create logs directory if it doesn't exist
    LOGS_DIR.mkdir(exist_ok=True)
    
    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = LOGS_DIR / f"batch_processing_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)  # Also output to console
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_filename}")
    return logger

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
    # Initialize logging
    logger = setup_logging()
    
    logging.info("=" * 60)
    logging.info("Starting batch processing session")
    logging.info(f"Script: {Path(__file__).name}")
    logging.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("=" * 60)
    
    object_ids = load_ids(IDS_PATH)
    object_ids = object_ids[START:]
    
    logging.info(f"Loaded {len(object_ids)} object IDs to process")
    logging.info(f"Starting from index {START}")

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
                logging.info(f"[skip-404] {oid}")
                i += 1            # advance past this oid
                continue
            logging.warning(f"[retry] {oid}: HTTP {getattr(e.response,'status_code',None)}. sleeping {BIG_DELAY}s")
            time.sleep(BIG_DELAY) # stay on same oid
            continue
        except requests.RequestException as e:
            logging.warning(f"[retry] {oid}: http error: {e}. sleeping {BIG_DELAY}s")
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
            logging.error(f"[retry] mongo error for {oid}: {e}. sleeping {BIG_DELAY}s")
            time.sleep(BIG_DELAY) # stay on same oid
            continue

        processed += 1
        logging.info(f"[ok] {START + processed} upserted {oid}")
        i += 1  # advance only after success or a 404 skip

    # Session end logging
    logging.info("=" * 60)
    logging.info("Batch processing session completed")
    logging.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Total processed: {processed}")
    logging.info(f"Total objects in batch: {n}")
    logging.info("=" * 60)

if __name__ == "__main__":
    main()