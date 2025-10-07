#!/usr/bin/env python3
"""
Script to filter documents from the Raw collection and output to Batch-2 collection.

Filtering criteria:
1. Document must contain a string in the field "data.primaryImage"
2. Document must have "data.classification" containing one of: "Paintings", "Paper-Paintings", "Paintings-Canvas"
3. Document must have "data.objectName" containing "painting" (case insensitive)

Usage: python filter_to_batch2.py
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from pymongo import MongoClient
from pymongo.errors import BulkWriteError

# MongoDB connection settings
MONGODB_URI = "mongodb+srv://admin:2I5At8MJUVPNJgmF@meana-lisa.xswknhv.mongodb.net/"
DB_NAME = "Paintings"
RAW_COLLECTION = "Raw"
BATCH2_COLLECTION = "Batch-2"

# Filter criteria
VALID_CLASSIFICATIONS = ["Paintings", "Paper-Paintings", "Paintings-Canvas"]
OBJECT_NAME_KEYWORD = "painting"

# Logging setup
LOGS_DIR = Path("logs")

def setup_logging():
    """Setup logging with timestamped file and console output."""
    # Create logs directory if it doesn't exist
    LOGS_DIR.mkdir(exist_ok=True)
    
    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = LOGS_DIR / f"batch2_filtering_{timestamp}.log"
    
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

def check_primary_image(doc: Dict[Any, Any]) -> bool:
    """Check if document has a non-empty primaryImage field."""
    try:
        primary_image = doc.get("data", {}).get("primaryImage", "")
        return bool(primary_image and isinstance(primary_image, str) and primary_image.strip())
    except (AttributeError, KeyError):
        return False

def check_classification(doc: Dict[Any, Any]) -> bool:
    """Check if document's classification is in the valid list."""
    try:
        classification = doc.get("data", {}).get("classification", "")
        return classification in VALID_CLASSIFICATIONS
    except (AttributeError, KeyError):
        return False

def check_object_name(doc: Dict[Any, Any]) -> bool:
    """Check if document's objectName contains 'painting' (case insensitive)."""
    try:
        object_name = doc.get("data", {}).get("objectName", "")
        return OBJECT_NAME_KEYWORD.lower() in object_name.lower()
    except (AttributeError, KeyError):
        return False

def filter_document(doc: Dict[Any, Any]) -> bool:
    """Apply all filter criteria to a document."""
    return (
        check_primary_image(doc) and
        check_classification(doc) and
        check_object_name(doc)
    )

def process_documents_batch(client: MongoClient, batch: List[Dict[Any, Any]], logger: logging.Logger) -> int:
    """Process a batch of filtered documents and insert into Batch-2 collection."""
    if not batch:
        return 0
    
    try:
        db = client[DB_NAME]
        batch2_collection = db[BATCH2_COLLECTION]
        
        # Insert the batch
        result = batch2_collection.insert_many(batch)
        logger.info(f"Successfully inserted {len(result.inserted_ids)} documents into Batch-2")
        return len(result.inserted_ids)
        
    except BulkWriteError as e:
        logger.error(f"Bulk write error: {e}")
        return 0
    except Exception as e:
        logger.error(f"Error inserting batch: {e}")
        return 0

def main():
    """Main function to filter and process documents."""
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("Starting Batch-2 filtering process")
    logger.info(f"Script: {Path(__file__).name}")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Valid classifications: {VALID_CLASSIFICATIONS}")
    logger.info(f"Object name keyword: '{OBJECT_NAME_KEYWORD}'")
    logger.info("=" * 60)
    
    try:
        # Connect to MongoDB
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        raw_collection = db[RAW_COLLECTION]
        
        # Get total count for progress tracking
        total_docs = raw_collection.count_documents({})
        logger.info(f"Total documents in Raw collection: {total_docs}")
        
        # Process documents in batches
        batch_size = 1000
        processed_count = 0
        filtered_count = 0
        inserted_count = 0
        batch = []
        
        # Get all documents from Raw collection
        cursor = raw_collection.find({})
        
        for doc in cursor:
            processed_count += 1
            
            # Apply filters
            if filter_document(doc):
                filtered_count += 1
                batch.append(doc)
                
                # Process batch when it reaches batch_size
                if len(batch) >= batch_size:
                    inserted = process_documents_batch(client, batch, logger)
                    inserted_count += inserted
                    batch = []
            
            # Log progress every 1000 documents
            if processed_count % 1000 == 0:
                logger.info(f"Processed: {processed_count}/{total_docs} | "
                          f"Filtered: {filtered_count} | Inserted: {inserted_count}")
        
        # Process remaining documents in the last batch
        if batch:
            inserted = process_documents_batch(client, batch, logger)
            inserted_count += inserted
        
        # Final statistics
        logger.info("=" * 60)
        logger.info("Batch-2 filtering process completed")
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total documents processed: {processed_count}")
        logger.info(f"Documents that passed all filters: {filtered_count}")
        logger.info(f"Documents inserted into Batch-2: {inserted_count}")
        logger.info(f"Filter success rate: {(filtered_count/processed_count*100):.2f}%" if processed_count > 0 else "N/A")
        logger.info("=" * 60)
        
        # Verify the new collection
        batch2_count = db[BATCH2_COLLECTION].count_documents({})
        logger.info(f"Final Batch-2 collection count: {batch2_count}")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        client.close()

if __name__ == "__main__":
    main()
