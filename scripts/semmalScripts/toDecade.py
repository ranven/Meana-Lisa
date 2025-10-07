#!/usr/bin/env python3
"""
Script to convert objectEndDate years to decades and centuries, then update MongoDB documents.

This script:
1. Converts years to decades (e.g., 1807 → 1800, 1923 → 1920)
2. Converts years to centuries (e.g., 1807 → 19, 1923 → 20)
3. Updates all documents in Batch-3 collection with objectDecade and objectCentury fields

Usage: python toDecade.py
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from pymongo import MongoClient
from pymongo.errors import BulkWriteError, OperationFailure

# MongoDB connection settings
MONGODB_URI = "mongodb+srv://admin:2I5At8MJUVPNJgmF@meana-lisa.xswknhv.mongodb.net/"
DB_NAME = "Paintings"
BATCH3_COLLECTION = "Batch-3"

# Logging setup
LOGS_DIR = Path("logs")

def setup_logging():
    """Setup logging with timestamped file and console output."""
    # Create logs directory if it doesn't exist
    LOGS_DIR.mkdir(exist_ok=True)
    
    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = LOGS_DIR / f"decade_conversion_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)  # Also output to console
        ]
    )
    
    return log_filename

def year_to_decade(year: int) -> int:
    """
    Convert year to decade.
    
    Args:
        year: The year to convert
        
    Returns:
        The decade (e.g., 1807 → 1800, 1923 → 1920)
    """
    return (year // 10) * 10

def year_to_century(year: int) -> int:
    """
    Convert year to century.
    
    Args:
        year: The year to convert
        
    Returns:
        The century (e.g., 1807 → 19, 1923 → 20, 2001 → 21)
    """
    return (year - 1) // 100 + 1

def update_documents_with_decades_and_centuries():
    """
    Update all documents in Batch-3 collection with objectDecade and objectCentury fields.
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Connect to MongoDB
        logger.info("Connecting to MongoDB...")
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        collection = db[BATCH3_COLLECTION]
        
        # Get total count of documents
        total_docs = collection.count_documents({})
        logger.info(f"Found {total_docs} documents in {BATCH3_COLLECTION} collection")
        
        if total_docs == 0:
            logger.warning("No documents found in collection")
            return
        
        # Find documents that have objectEndDate but don't have objectDecade or objectCentury
        query = {
            "objectEndDate": {"$exists": True, "$ne": None},
            "$or": [
                {"objectDecade": {"$exists": False}},
                {"objectCentury": {"$exists": False}}
            ]
        }
        
        docs_to_update = collection.find(query)
        docs_count = collection.count_documents(query)
        logger.info(f"Found {docs_count} documents that need updating")
        
        if docs_count == 0:
            logger.info("All documents already have decade and century fields")
            return
        
        # Process documents in batches
        batch_size = 1000
        processed = 0
        updated = 0
        errors = 0
        
        logger.info(f"Starting batch processing with batch size {batch_size}")
        
        for doc in docs_to_update:
            try:
                object_end_date = doc.get('objectEndDate')
                
                # Skip if objectEndDate is not a valid integer
                if not isinstance(object_end_date, int):
                    logger.warning(f"Document {doc.get('_id')} has non-integer objectEndDate: {object_end_date}")
                    continue
                
                # Calculate decade and century
                decade = year_to_decade(object_end_date)
                century = year_to_century(object_end_date)
                
                # Update the document
                update_result = collection.update_one(
                    {"_id": doc["_id"]},
                    {
                        "$set": {
                            "objectDecade": decade,
                            "objectCentury": century
                        }
                    }
                )
                
                if update_result.modified_count > 0:
                    updated += 1
                    logger.debug(f"Updated document {doc['_id']}: {object_end_date} → decade: {decade}, century: {century}")
                
                processed += 1
                
                # Log progress every 100 documents
                if processed % 100 == 0:
                    logger.info(f"Processed {processed}/{docs_count} documents...")
                
            except Exception as e:
                errors += 1
                logger.error(f"Error processing document {doc.get('_id', 'unknown')}: {str(e)}")
                continue
        
        logger.info(f"Processing complete!")
        logger.info(f"Total processed: {processed}")
        logger.info(f"Successfully updated: {updated}")
        logger.info(f"Errors: {errors}")
        
        # Verify the update by checking a few sample documents
        logger.info("Verifying updates...")
        sample_docs = collection.find({"objectDecade": {"$exists": True}}).limit(5)
        for doc in sample_docs:
            logger.info(f"Sample document {doc['_id']}: objectEndDate={doc.get('objectEndDate')}, "
                       f"objectDecade={doc.get('objectDecade')}, objectCentury={doc.get('objectCentury')}")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise
    finally:
        if 'client' in locals():
            client.close()
            logger.info("MongoDB connection closed")

def main():
    """Main function to run the decade conversion process."""
    log_file = setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("Starting decade and century conversion process")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 60)
    
    try:
        update_documents_with_decades_and_centuries()
        logger.info("Decade and century conversion completed successfully!")
    except Exception as e:
        logger.error(f"Process failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

