#!/usr/bin/env python3
"""
Script to process documents from Batch-2 collection and create Batch-3 collection with color palettes.

For each document in Batch-2:
1. Extract the primaryImage URL
2. Process the image to extract color palette using K-means clustering
3. Create a new document in Batch-3 with selected fields plus the color palette

Usage: python batch2_to_batch3.py
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import time
import warnings

from pymongo import MongoClient
from pymongo.errors import BulkWriteError, ConnectionFailure
from skimage.io import imread
from skimage.transform import resize
from sklearn.cluster import KMeans
import numpy as np
from urllib.parse import urlparse
import requests
from requests.exceptions import RequestException, Timeout

# Suppress specific runtime warnings from scikit-learn
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*divide by zero.*")
warnings.filterwarnings("ignore", message=".*overflow.*")
warnings.filterwarnings("ignore", message=".*invalid value.*")

# MongoDB connection settings
MONGODB_URI = "mongodb+srv://admin:2I5At8MJUVPNJgmF@meana-lisa.xswknhv.mongodb.net/"
DB_NAME = "Paintings"
BATCH2_COLLECTION = "Batch-2"
BATCH3_COLLECTION = "Batch-3"

# Image processing settings
KMEANS_CLUSTERS = 10  # Reduced from 20 for faster processing
KMEANS_RANDOM_STATE = 0
IMAGE_TIMEOUT = 120  # seconds
MAX_RETRIES = 3
MAX_IMAGE_SIZE = 300  # Resize images to max 300x300 pixels

# Processing configuration - EDIT THESE VALUES
START_FROM_DOCUMENT = 1  # Document number to start from (1-based index)
PROCESSING_LIMIT = None   # Maximum number of documents to process (None = process all)

# Logging setup
LOGS_DIR = Path("logs")

def setup_logging():
    """Setup logging with timestamped file and console output."""
    # Create logs directory if it doesn't exist
    LOGS_DIR.mkdir(exist_ok=True)
    
    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = LOGS_DIR / f"batch2_to_batch3_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)  # Also output to console
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_filename}")
    return logger


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Convert RGB tuple to hex color string."""
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

def extract_color_palette(image_url: str, logger: logging.Logger) -> Optional[List[Tuple[str, float]]]:
    """
    Extract color palette from image URL using optimized K-means clustering.
    
    Args:
        image_url: URL of the image to process
        logger: Logger instance for error reporting
        
    Returns:
        List of tuples (hex_color, percentage) sorted by percentage (highest first)
        None if processing fails
    """
    try:
        # Validate URL
        parsed_url = urlparse(image_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            logger.warning(f"Invalid URL format: {image_url}")
            return None
        
        logger.info(f"Processing image: {image_url}")
        
        # Load image with timeout handling
        try:
            # Use requests to download with timeout, then load with skimage
            response = requests.get(image_url, timeout=IMAGE_TIMEOUT, stream=True)
            response.raise_for_status()
            
            # Load image from the response content
            from io import BytesIO
            img = imread(BytesIO(response.content))
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout loading image {image_url} (>{IMAGE_TIMEOUT}s)")
            return None
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error loading image {image_url}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to load image {image_url}: {e}")
            return None
        
        # Check if image loaded successfully
        if img is None or img.size == 0:
            logger.warning(f"Failed to load image: {image_url}")
            return None
        
        logger.info(f"Original image shape: {img.shape}")
        
        # Resize image to max 300x300 pixels for much faster processing
        h, w = img.shape[:2]
        if h > w:
            new_h, new_w = MAX_IMAGE_SIZE, int(w * MAX_IMAGE_SIZE / h)
        else:
            new_h, new_w = int(h * MAX_IMAGE_SIZE / w), MAX_IMAGE_SIZE
        
        img_resized = resize(img, (new_h, new_w), anti_aliasing=True)
        img_resized = (img_resized * 255).astype(np.uint8)
        logger.info(f"Resized image shape: {img_resized.shape}")
        
        # Sample pixels instead of using all pixels (every 4th pixel)
        img_sampled = img_resized[::4, ::4]
        logger.info(f"Sampled image shape: {img_sampled.shape}")
        
        img_flat = img_sampled.reshape((-1, 3))
        logger.info(f"Total pixels to process: {len(img_flat)}")
        
        # Apply K-means clustering with reduced clusters
        logger.info("Running K-means clustering...")
        kmeans = KMeans(n_clusters=KMEANS_CLUSTERS, random_state=KMEANS_RANDOM_STATE, n_init=10)
        kmeans.fit(img_flat)
        
        # Get cluster centers (dominant colors)
        palette = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        
        # Convert RGB to hex
        hex_palette = [rgb_to_hex(color) for color in palette]
        
        # Calculate weights (percentages) for each color
        total_pixels = len(labels)
        color_counts = np.bincount(labels)
        color_percentages = (color_counts / total_pixels) * 100
        
        # Create array of tuples (hex_color, percentage) and sort by percentage (highest first)
        color_distribution = [(hex_color, float(percentage)) for hex_color, percentage in zip(hex_palette, color_percentages)]
        color_distribution.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Successfully extracted {len(color_distribution)} colors from image")
        return color_distribution
        
    except Exception as e:
        logger.error(f"Error processing image {image_url}: {e}")
        return None

def extract_document_fields(doc: Dict[Any, Any]) -> Dict[str, Any]:
    """
    Extract required fields from Batch-2 document.
    
    Args:
        doc: Document from Batch-2 collection
        
    Returns:
        Dictionary with extracted fields
    """
    data = doc.get("data", {})
    
    return {
        "objectID": data.get("objectID"),
        "isHighlight": data.get("isHighlight"),
        "primaryImage": data.get("primaryImage"),
        "department": data.get("department"),
        "objectName": data.get("objectName"),
        "title": data.get("title"),
        "artistDisplayName": data.get("artistDisplayName"),
        "artistNationality": data.get("artistNationality"),
        "artistBeginDate": data.get("artistBeginDate"),
        "artistEndDate": data.get("artistEndDate"),
        "artistWikidata_URL": data.get("artistWikidata_URL"),
        "objectBeginDate": data.get("objectBeginDate"),
        "objectEndDate": data.get("objectEndDate"),
        "medium": data.get("medium"),
        "dimensions": data.get("dimensions"),
        "classification": data.get("classification"),
        "objectURL": data.get("objectURL")
    }

def process_single_document(doc: Dict[Any, Any], client: MongoClient, logger: logging.Logger) -> Tuple[bool, str]:
    """
    Process a single document from Batch-2 and insert into Batch-3.
    
    Args:
        doc: Document from Batch-2 collection
        client: MongoDB client
        logger: Logger instance
        
    Returns:
        Tuple of (success: bool, object_id: str)
    """
    object_id = str(doc.get('_id', 'unknown'))
    try:
        # Extract primary image URL
        primary_image = doc.get("data", {}).get("primaryImage")
        if not primary_image:
            logger.warning(f"Document {object_id} has no primaryImage")
            return False, object_id
        
        # Extract color palette
        palette = extract_color_palette(primary_image, logger)
        if palette is None:
            logger.warning(f"Failed to extract palette for document {object_id}")
            return False, object_id
        
        # Extract required fields
        extracted_fields = extract_document_fields(doc)
        
        # Extract primary color (first tuple in palette - most dominant color)
        primary_colour = palette[0][0] if palette else None
        
        # Create new document for Batch-3
        new_doc = {
            **extracted_fields,
            "palette": palette,
            "primaryColour": primary_colour,
            "processed_at": datetime.now().isoformat(),
            "source_object_id": doc.get("_id")
        }
        
        # MongoDB operations
        db = client[DB_NAME]
        batch3_collection = db[BATCH3_COLLECTION]
        
        # Check if document with same objectID already exists
        existing_doc = batch3_collection.find_one({"objectID": new_doc["objectID"]})
        
        if existing_doc:
            # Update existing document
            result = batch3_collection.update_one(
                {"objectID": new_doc["objectID"]}, 
                {"$set": new_doc}
            )
            if result.modified_count > 0:
                logger.info(f"Successfully updated document {object_id} (objectID: {new_doc['objectID']})")
            else:
                logger.warning(f"No changes made to document {object_id} (objectID: {new_doc['objectID']})")
        else:
            # Insert new document
            result = batch3_collection.insert_one(new_doc)
            logger.info(f"Successfully created document {object_id} -> {result.inserted_id} (objectID: {new_doc['objectID']})")
        
        return True, object_id
        
    except Exception as e:
        logger.error(f"Error processing document {object_id}: {e}")
        return False, object_id


def main():
    """Main function to process documents from Batch-2 to Batch-3."""
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("Starting Batch-2 to Batch-3 processing")
    logger.info(f"Script: {Path(__file__).name}")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"K-means clusters: {KMEANS_CLUSTERS}")
    logger.info(f"Image timeout: {IMAGE_TIMEOUT}s")
    logger.info(f"Max image size: {MAX_IMAGE_SIZE}x{MAX_IMAGE_SIZE}")
    logger.info(f"Starting from document: {START_FROM_DOCUMENT}")
    if PROCESSING_LIMIT:
        logger.info(f"Processing limit: {PROCESSING_LIMIT} documents")
    logger.info("=" * 60)
    
    try:
        # Connect to MongoDB
        logger.info("Connecting to MongoDB...")
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        
        # Test connection
        client.admin.command('ping')
        logger.info("Successfully connected to MongoDB")
        
        db = client[DB_NAME]
        batch2_collection = db[BATCH2_COLLECTION]
        
        # Get total count for progress tracking
        total_docs = batch2_collection.count_documents({})
        logger.info(f"Total documents in Batch-2 collection: {total_docs}")
        
        if total_docs == 0:
            logger.warning("No documents found in Batch-2 collection")
            return
        
        # Get all documents from Batch-2 collection
        all_docs = list(batch2_collection.find({}))
        
        # Apply start position and limit filters
        docs_to_process = all_docs[START_FROM_DOCUMENT - 1:]
        if PROCESSING_LIMIT:
            docs_to_process = docs_to_process[:PROCESSING_LIMIT]
        
        logger.info(f"Documents to process: {len(docs_to_process)}")
        
        # Process documents sequentially
        total_successful = 0
        total_failed = 0
        
        logger.info(f"Starting sequential processing of {len(docs_to_process)} documents")
        
        for i, doc in enumerate(docs_to_process, 1):
            logger.info(f"Processing document {i}/{len(docs_to_process)}")
            doc_start_time = time.time()
            
            # Process the document
            success, object_id = process_single_document(doc, client, logger)
            
            if success:
                total_successful += 1
            else:
                total_failed += 1
            
            doc_end_time = time.time()
            doc_duration = doc_end_time - doc_start_time
            
            logger.info(f"Document {i} completed in {doc_duration:.2f}s")
            logger.info(f"Progress: {i}/{len(docs_to_process)} | "
                       f"Successful: {total_successful} | Failed: {total_failed}")
            
            # Small delay between documents to avoid overwhelming the system
            if i < len(docs_to_process):
                time.sleep(0.5)
        
        # Final statistics
        logger.info("=" * 60)
        logger.info("Batch-2 to Batch-3 processing completed")
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total documents in collection: {total_docs}")
        logger.info(f"Documents skipped: {START_FROM_DOCUMENT - 1}")
        logger.info(f"Documents processed: {total_successful + total_failed}")
        logger.info(f"Successfully processed: {total_successful}")
        logger.info(f"Failed to process: {total_failed}")
        logger.info(f"Success rate: {(total_successful/(total_successful + total_failed)*100):.2f}%" if (total_successful + total_failed) > 0 else "N/A")
        logger.info("=" * 60)
        
        # Verify the new collection
        batch3_count = db[BATCH3_COLLECTION].count_documents({})
        logger.info(f"Final Batch-3 collection count: {batch3_count}")
        
    except ConnectionFailure as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        client.close()

if __name__ == "__main__":
    main()
