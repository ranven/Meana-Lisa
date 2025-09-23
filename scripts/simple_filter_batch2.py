#!/usr/bin/env python3
"""
Simple script to filter documents from Raw collection and insert into Batch-2 collection.

Filtering criteria:
1. Document must contain a string in the field "data.primaryImage"
2. Document must have "data.classification" containing one of: "Paintings", "Paper-Paintings", "Paintings-Canvas"
3. Document must have "data.objectName" containing "painting" (case insensitive)
"""

import logging
from pymongo import MongoClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('batch2.log')
    ],
    force=True  # Force reconfiguration
)
logger = logging.getLogger(__name__)

# MongoDB connection
client = MongoClient("mongodb+srv://admin:2I5At8MJUVPNJgmF@meana-lisa.xswknhv.mongodb.net/")
db = client["Paintings"]
raw_collection = db["Raw"]
batch2_collection = db["Batch-2"]

# Valid classifications
valid_classifications = ["Paintings", "Paper-Paintings", "Paintings-Canvas"]

logger.info("Starting to filter documents...")

# Counter for tracking
total_processed = 0
filtered_count = 0

# Loop through all documents in Raw collection
for doc in raw_collection.find({}):
    total_processed += 1
    
    # Get the data object
    data = doc.get("data", {})
    doc_id = doc.get("_id", "unknown")
    
    # Check 1: primaryImage exists and is not empty
    primary_image = data.get("primaryImage", "")
    if not primary_image or not isinstance(primary_image, str) or not primary_image.strip():
        logger.info(f"ID {doc_id}: EXCLUDED - No primaryImage")
        continue
    
    # Check 2: classification logic
    classification = data.get("classification", "")
    object_name = data.get("objectName", "")
    
    if classification:  # If classification is not empty
        # Check if classification is in valid list
        if classification not in valid_classifications:
            logger.info(f"ID {doc_id}: EXCLUDED - Classification '{classification}' not in valid list")
            continue
        # If classification is valid, continue (no objectName check needed)
    else:  # If classification is empty
        # Only check objectName contains "painting" (case insensitive)
        if "painting" not in object_name.lower():
            logger.info(f"ID {doc_id}: EXCLUDED - ObjectName '{object_name}' doesn't contain 'painting'")
            continue
    
    # If all checks pass, insert into Batch-2
    batch2_collection.insert_one(doc)
    filtered_count += 1
    logger.info(f"ID {doc_id}: INCLUDED - Passed all filters")
    
    # Log progress every 1000 documents
    if total_processed % 1000 == 0:
        logger.info(f"--- Progress: Processed {total_processed}, Filtered {filtered_count} ---")

logger.info("Completed!")
logger.info(f"Total documents processed: {total_processed}")
logger.info(f"Documents that passed all filters: {filtered_count}")
logger.info(f"Documents in Batch-2 collection: {batch2_collection.count_documents({})}")

client.close()
