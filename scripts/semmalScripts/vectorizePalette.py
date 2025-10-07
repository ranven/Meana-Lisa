import pymongo
from datetime import datetime
import logging

# MongoDB connection settings
MONGODB_URI = "mongodb+srv://admin:2I5At8MJUVPNJgmF@meana-lisa.xswknhv.mongodb.net/"
DB_NAME = "Paintings"
BATCH3_COLLECTION = "Batch-3"

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def vectorize_palette(palette, scaling_factor=10.0):
    """
    Convert palette array to 15-dimensional vector (first 5 colors only)
    with enhanced scaling to make color differences more prominent
    
    Args:
        palette: List of [hex_color, weight] pairs
        scaling_factor: Multiplier to make color differences more dramatic (default: 10.0)
    
    Returns:
        List of 15 numbers representing weighted RGB values for first 5 colors
    """
    vector = []
    
    # Process only the first 5 colors, pad with zeros if needed
    for i in range(5):
        if i < len(palette) and len(palette[i]) == 2:
            hex_color, weight = palette[i]
            r, g, b = hex_to_rgb(hex_color)
            
            # Apply weight to RGB values and scale for more dramatic differences
            vector.extend([r * weight * scaling_factor, g * weight * scaling_factor, b * weight * scaling_factor])
        else:
            # Pad with zeros if palette has fewer than 5 colors
            vector.extend([0.0, 0.0, 0.0])
    
    return vector

def process_batch3_collection():
    """Process all documents in batch3 collection and add vectorizedPalette field"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'vectorize_palette_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    
    try:
        # Connect to MongoDB
        client = pymongo.MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        collection = db[BATCH3_COLLECTION]
        
        logging.info(f"Connected to MongoDB. Processing collection: {BATCH3_COLLECTION}")
        
        # Get total count for progress tracking
        total_docs = collection.count_documents({})
        logging.info(f"Total documents to process: {total_docs}")
        
        processed_count = 0
        error_count = 0
        
        # Process each document
        for document in collection.find({}):
            try:
                doc_id = document['_id']
                
                # Note: We're updating all documents, including those with existing vectorizedPalette
                
                # Check if document has palette field
                if 'palette' not in document:
                    logging.warning(f"Document {doc_id} has no palette field, skipping")
                    error_count += 1
                    continue
                
                palette = document['palette']
                
                # Validate palette structure
                if not isinstance(palette, list):
                    logging.warning(f"Document {doc_id} has invalid palette format, skipping")
                    error_count += 1
                    continue
                
                # Vectorize the palette
                vectorized_palette = vectorize_palette(palette)
                
                # Update the document
                collection.update_one(
                    {'_id': doc_id},
                    {'$set': {'vectorizedPalette': vectorized_palette}}
                )
                
                processed_count += 1
                
                # Log progress every 100 documents
                if processed_count % 100 == 0:
                    logging.info(f"Processed {processed_count}/{total_docs} documents")
                
            except Exception as e:
                logging.error(f"Error processing document {document.get('_id', 'unknown')}: {str(e)}")
                error_count += 1
                continue
        
        logging.info(f"Processing complete!")
        logging.info(f"Successfully processed: {processed_count} documents")
        logging.info(f"Errors encountered: {error_count} documents")
        
    except Exception as e:
        logging.error(f"Database connection error: {str(e)}")
    finally:
        client.close()

if __name__ == "__main__":
    process_batch3_collection()