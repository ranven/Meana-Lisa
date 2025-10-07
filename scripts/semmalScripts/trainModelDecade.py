#!/usr/bin/env python3
"""
Script to train a logistic regression model for predicting decades from vectorized palettes.

This script:
1. Connects to MongoDB and extracts vectorizedPalette and objectDecade data
2. Prepares feature matrix X (n_samples × VECTORIZED_PALETTE_LENGTH) and labels y (decades)
3. Implements train-test split and feature scaling
4. Trains logistic regression model with multinomial classification
5. Evaluates model performance and saves the trained model

Usage: python trainModelDecade.py
"""

import logging
import sys
import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pymongo
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Configuration settings
VECTORIZED_PALETTE_LENGTH = 15  # Expected length of vectorizedPalette arrays (was 30, now 15)

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
    log_filename = LOGS_DIR / f"model_training_{timestamp}.log"
    
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

def extract_training_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract vectorizedPalette and objectDecade data from MongoDB.
    Filters out decades with fewer than 2 samples to avoid stratified split errors.
    
    Returns:
        Tuple of (X, y) where:
        - X: Feature matrix (n_samples × VECTORIZED_PALETTE_LENGTH) of vectorized palettes
        - y: Labels array (n_samples) of decades
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Connect to MongoDB
        logger.info("Connecting to MongoDB...")
        client = pymongo.MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        collection = db[BATCH3_COLLECTION]
        
        # Query for documents that have both vectorizedPalette and objectDecade
        query = {
            "vectorizedPalette": {"$exists": True},
            "objectDecade": {"$exists": True, "$ne": None}
        }
        
        # Get total count for progress tracking
        total_docs = collection.count_documents(query)
        logger.info(f"Found {total_docs} documents with both vectorizedPalette and objectDecade")
        
        if total_docs == 0:
            logger.error("No documents found with required fields")
            return np.array([]), np.array([])
        
        # Extract data
        X = []  # features (vectorized palettes)
        y = []  # labels (decades)
        
        processed = 0
        skipped = 0
        
        for doc in collection.find(query):
            try:
                vectorized_palette = doc.get("vectorizedPalette")
                object_decade = doc.get("objectDecade")
                
                # Validate data
                if not isinstance(vectorized_palette, list) or len(vectorized_palette) != VECTORIZED_PALETTE_LENGTH:
                    logger.warning(f"Document {doc['_id']} has invalid vectorizedPalette format, skipping")
                    skipped += 1
                    continue
                
                if not isinstance(object_decade, int):
                    logger.warning(f"Document {doc['_id']} has invalid objectDecade format, skipping")
                    skipped += 1
                    continue
                
                # Convert to numpy arrays and add to training data
                X.append(vectorized_palette)
                y.append(object_decade)
                
                processed += 1
                
                # Log progress every 1000 documents
                if processed % 1000 == 0:
                    logger.info(f"Processed {processed}/{total_docs} documents...")
                
            except Exception as e:
                logger.error(f"Error processing document {doc.get('_id', 'unknown')}: {str(e)}")
                skipped += 1
                continue
        
        logger.info(f"Data extraction complete!")
        logger.info(f"Successfully processed: {processed} documents")
        logger.info(f"Skipped: {skipped} documents")
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Labels shape: {y.shape}")
        
        # Analyze decade distribution and filter out rare decades
        unique_decades, counts = np.unique(y, return_counts=True)
        logger.info(f"Decade distribution:")
        for decade, count in zip(unique_decades, counts):
            logger.info(f"  {decade}: {count} samples")
        
        # Filter out decades with fewer than 2 samples
        min_samples_per_class = 2
        valid_decades = unique_decades[counts >= min_samples_per_class]
        filtered_decades = unique_decades[counts < min_samples_per_class]
        
        if len(filtered_decades) > 0:
            logger.warning(f"Filtering out decades with < {min_samples_per_class} samples: {filtered_decades.tolist()}")
            logger.warning(f"Total samples being filtered out: {sum(counts[counts < min_samples_per_class])}")
        
        # Create mask for valid samples
        valid_mask = np.isin(y, valid_decades)
        X_filtered = X[valid_mask]
        y_filtered = y[valid_mask]
        
        logger.info(f"After filtering:")
        logger.info(f"  Feature matrix shape: {X_filtered.shape}")
        logger.info(f"  Labels shape: {y_filtered.shape}")
        logger.info(f"  Valid decades: {sorted(valid_decades.tolist())}")
        logger.info(f"  Number of valid decades: {len(valid_decades)}")
        
        if len(valid_decades) < 2:
            logger.error(f"Not enough valid decades ({len(valid_decades)}) for classification. Need at least 2.")
            return np.array([]), np.array([])
        
        return X_filtered, y_filtered
        
    except Exception as e:
        logger.error(f"Error extracting training data: {str(e)}")
        raise
    finally:
        if 'client' in locals():
            client.close()
            logger.info("MongoDB connection closed")

def train_logistic_regression_model(X: np.ndarray, y: np.ndarray) -> Tuple[LogisticRegression, StandardScaler, LabelEncoder]:
    """
    Train logistic regression model for decade prediction.
    
    Args:
        X: Feature matrix (n_samples × VECTORIZED_PALETTE_LENGTH)
        y: Labels array (n_samples)
        
    Returns:
        Tuple of (trained_model, scaler, label_encoder)
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting model training...")
        
        # Encode decades (convert decade values to integer labels)
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        
        logger.info(f"Encoded labels shape: {y_encoded.shape}")
        logger.info(f"Number of unique classes: {len(encoder.classes_)}")
        logger.info(f"Class labels: {encoder.classes_}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        logger.info(f"Training set size: {X_train.shape[0]}")
        logger.info(f"Test set size: {X_test.shape[0]}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        logger.info("Features scaled using StandardScaler")
        
        # Train multinomial logistic regression
        clf = LogisticRegression(
            multi_class='multinomial',
            max_iter=1000,
            random_state=42,
            solver='lbfgs'  # Good for small datasets
        )
        
        logger.info("Training logistic regression model...")
        clf.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_score = clf.score(X_train_scaled, y_train)
        test_score = clf.score(X_test_scaled, y_test)
        
        logger.info(f"Training accuracy: {train_score:.4f}")
        logger.info(f"Test accuracy: {test_score:.4f}")
        
        # Generate predictions for detailed evaluation
        y_pred = clf.predict(X_test_scaled)
        
        # Get unique classes that appear in test set
        test_classes = np.unique(y_test)
        logger.info(f"Classes present in test set: {len(test_classes)}")
        logger.info(f"Classes present in training set: {len(encoder.classes_)}")
        
        # Classification report - only include classes that appear in test set
        logger.info("Classification Report:")
        logger.info(f"\n{classification_report(y_test, y_pred, labels=test_classes, target_names=encoder.inverse_transform(test_classes).astype(str))}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=test_classes)
        logger.info("Confusion Matrix:")
        logger.info(f"\n{cm}")
        
        return clf, scaler, encoder
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

def save_model_and_preprocessors(model: LogisticRegression, scaler: StandardScaler, encoder: LabelEncoder):
    """
    Save the trained model and preprocessors to files.
    
    Args:
        model: Trained logistic regression model
        scaler: Fitted StandardScaler
        encoder: Fitted LabelEncoder
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Create models directory if it doesn't exist
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = models_dir / f"decade_predictor_model_{timestamp}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to: {model_path}")
        
        # Save scaler
        scaler_path = models_dir / f"decade_predictor_scaler_{timestamp}.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f"Scaler saved to: {scaler_path}")
        
        # Save encoder
        encoder_path = models_dir / f"decade_predictor_encoder_{timestamp}.pkl"
        with open(encoder_path, 'wb') as f:
            pickle.dump(encoder, f)
        logger.info(f"Encoder saved to: {encoder_path}")
        
        # Save metadata
        metadata = {
            "timestamp": timestamp,
            "model_type": "LogisticRegression",
            "feature_dimension": VECTORIZED_PALETTE_LENGTH,
            "classes": encoder.classes_.tolist(),
            "n_classes": len(encoder.classes_),
            "model_path": str(model_path),
            "scaler_path": str(scaler_path),
            "encoder_path": str(encoder_path)
        }
        
        metadata_path = models_dir / f"decade_predictor_metadata_{timestamp}.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        logger.info(f"Metadata saved to: {metadata_path}")
        
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def vectorize_palette(palette: List[List], scaling_factor: float = 10.0) -> List[float]:
    """
    Convert palette array to vectorized format with enhanced scaling
    
    Args:
        palette: List of [hex_color, weight] pairs (uses first 5 colors)
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

def predict_decade_from_raw_palette(palette: List[List], model: LogisticRegression, 
                                  scaler: StandardScaler, encoder: LabelEncoder) -> Tuple[int, float]:
    """
    Predict decade from raw palette format (list of [hex_color, weight] pairs).
    
    Args:
        palette: List of [hex_color, weight] pairs (e.g., [["#ff0000", 0.5], ["#00ff00", 0.3]])
        model: Trained logistic regression model
        scaler: Fitted StandardScaler
        encoder: Fitted LabelEncoder
        
    Returns:
        Tuple of (predicted_decade, confidence_score)
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Convert raw palette to vectorized format
        vectorized_palette = vectorize_palette(palette)
        
        # Predict using the vectorized palette
        return predict_decade_from_palette(vectorized_palette, model, scaler, encoder)
        
    except Exception as e:
        logger.error(f"Error predicting decade from raw palette: {str(e)}")
        raise

def predict_decade_from_palette(vectorized_palette: List[float], model: LogisticRegression, 
                               scaler: StandardScaler, encoder: LabelEncoder) -> Tuple[int, float]:
    """
    Predict decade from a vectorized palette.
    
    Args:
        vectorized_palette: List of VECTORIZED_PALETTE_LENGTH numbers representing the vectorized palette
        model: Trained logistic regression model
        scaler: Fitted StandardScaler
        encoder: Fitted LabelEncoder
        
    Returns:
        Tuple of (predicted_decade, confidence_score)
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Convert to numpy array and reshape
        X = np.array(vectorized_palette).reshape(1, -1)
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Predict
        prediction = model.predict(X_scaled)[0]
        confidence = model.predict_proba(X_scaled).max()
        
        # Convert back to decade
        predicted_decade = encoder.inverse_transform([prediction])[0]
        
        return predicted_decade, confidence
        
    except Exception as e:
        logger.error(f"Error predicting decade: {str(e)}")
        raise

def main():
    """Main function to run the model training process."""
    log_file = setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("Starting decade prediction model training")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 60)
    
    try:
        # Extract training data
        logger.info("Step 1: Extracting training data from MongoDB...")
        X, y = extract_training_data()
        
        if len(X) == 0:
            logger.error("No training data available. Exiting.")
            sys.exit(1)
        
        # Train model
        logger.info("Step 2: Training logistic regression model...")
        model, scaler, encoder = train_logistic_regression_model(X, y)
        
        # Save model and preprocessors
        logger.info("Step 3: Saving trained model and preprocessors...")
        save_model_and_preprocessors(model, scaler, encoder)
        
        logger.info("=" * 60)
        logger.info("Model training completed successfully!")
        logger.info("=" * 60)
        
        # Test prediction with a sample
        logger.info("Testing prediction with first sample...")
        sample_palette = X[0].tolist()
        predicted_decade, confidence = predict_decade_from_palette(sample_palette, model, scaler, encoder)
        actual_decade = y[0]
        
        logger.info(f"Sample prediction:")
        logger.info(f"  Actual decade: {actual_decade}")
        logger.info(f"  Predicted decade: {predicted_decade}")
        logger.info(f"  Confidence: {confidence:.4f}")
        
    except Exception as e:
        logger.error(f"Process failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
