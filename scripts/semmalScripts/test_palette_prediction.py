#!/usr/bin/env python3
"""
Test script to predict decade from a raw palette.
This demonstrates how to use the trained model with your palette data.
"""

import pickle
import logging
from pathlib import Path

# Your palette data
test_palette = [
    ["#e3e0d5", 27.830303030303032],
    ["#e8e5dd", 23.684848484848484],
    ["#dddacc", 14.351515151515152],
    ["#3f3e3b", 12.096969696969696],
    ["#d5d1c0", 6.254545454545454],
    ["#353433", 6.133333333333333],
    ["#4c4945", 5.4787878787878785],
    ["#6a5e50", 2.0363636363636366],
    ["#90806c", 1.2363636363636363],
    ["#b6b2a9", 0.896969696969697]
]

def hex_to_rgb(hex_color: str):
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def vectorize_palette(palette, scaling_factor=10.0):
    """
    Convert palette array to 15-dimensional vector (first 5 colors only)
    with enhanced scaling to make color differences more prominent
    
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

def predict_decade_from_raw_palette(palette, model, scaler, encoder):
    """
    Predict decade from raw palette format (list of [hex_color, weight] pairs).
    """
    import numpy as np
    
    # Convert raw palette to vectorized format
    vectorized_palette = vectorize_palette(palette)
    
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

def main():
    """Test the prediction with your palette"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Look for the most recent model files
        models_dir = Path("models")
        if not models_dir.exists():
            logger.error("Models directory not found. Please run trainModelDecade.py first.")
            return
        
        # Find the most recent model files
        model_files = list(models_dir.glob("decade_predictor_model_*.pkl"))
        if not model_files:
            logger.error("No trained model found. Please run trainModelDecade.py first.")
            return
        
        # Get the most recent model
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        timestamp = latest_model.stem.split('_')[-2] + '_' + latest_model.stem.split('_')[-1]
        
        logger.info(f"Latest model: {latest_model}")
        logger.info(f"Extracted timestamp: {timestamp}")
        
        scaler_file = models_dir / f"decade_predictor_scaler_{timestamp}.pkl"
        encoder_file = models_dir / f"decade_predictor_encoder_{timestamp}.pkl"
        
        logger.info(f"Looking for scaler: {scaler_file}")
        logger.info(f"Looking for encoder: {encoder_file}")
        
        # Verify files exist
        if not scaler_file.exists():
            logger.error(f"Scaler file not found: {scaler_file}")
            return
        if not encoder_file.exists():
            logger.error(f"Encoder file not found: {encoder_file}")
            return
        
        # Load the model and preprocessors
        logger.info(f"Loading model from: {latest_model}")
        with open(latest_model, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Loading scaler from: {scaler_file}")
        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)
        
        logger.info(f"Loading encoder from: {encoder_file}")
        with open(encoder_file, 'rb') as f:
            encoder = pickle.load(f)
        
        # Test with your palette
        logger.info("Testing prediction with your palette...")
        logger.info("Palette colors:")
        for i, (color, weight) in enumerate(test_palette):
            logger.info(f"  {i+1}. {color} (weight: {weight:.2f})")
        
        # Make prediction
        predicted_decade, confidence = predict_decade_from_raw_palette(
            test_palette, model, scaler, encoder
        )
        
        logger.info("=" * 50)
        logger.info("PREDICTION RESULT:")
        logger.info(f"Predicted Decade: {predicted_decade}")
        logger.info(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        logger.info("=" * 50)
        
        # Show what the vectorized palette looks like
        vectorized = vectorize_palette(test_palette)
        logger.info(f"Vectorized palette (first 10 values): {vectorized[:10]}")
        logger.info(f"Vectorized palette length: {len(vectorized)}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
