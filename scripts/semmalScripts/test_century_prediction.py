#!/usr/bin/env python3
"""
Test script to predict century from a raw palette.
This demonstrates how to use the trained century model with your palette data.
"""

import pickle
import logging
from pathlib import Path

# Your palette data
test_palette = [
    [
      "#1a1511",
      42
    ],
    [
      "#231c17",
      13.214285714285715
    ],
    [
      "#4e3a24",
      11.214285714285714
    ],
    [
      "#312419",
      9.452380952380953
    ],
    [
      "#010000",
      9.11904761904762
    ],
    [
      "#3f2f1e",
      7.5
    ],
    [
      "#5b442b",
      3.3095238095238093
    ],
    [
      "#ab8c65",
      1.5
    ],
    [
      "#785b3c",
      1.4285714285714286
    ],
    [
      "#947451",
      1.2619047619047619
    ]
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

def predict_century_from_raw_palette(palette, model, scaler, encoder):
    """
    Predict century from raw palette format (list of [hex_color, weight] pairs).
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
    
    # Convert back to century
    predicted_century = encoder.inverse_transform([prediction])[0]
    
    return predicted_century, confidence

def main():
    """Test the prediction with your palette"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Look for the most recent century model files
        models_dir = Path("models")
        if not models_dir.exists():
            logger.error("Models directory not found. Please run trainModelCentury.py first.")
            return
        
        # Find the most recent century model files
        model_files = list(models_dir.glob("century_predictor_model_*.pkl"))
        if not model_files:
            logger.error("No trained century model found. Please run trainModelCentury.py first.")
            return
        
        # Get the most recent model
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        timestamp = latest_model.stem.split('_')[-2] + '_' + latest_model.stem.split('_')[-1]
        
        logger.info(f"Latest century model: {latest_model}")
        logger.info(f"Extracted timestamp: {timestamp}")
        
        scaler_file = models_dir / f"century_predictor_scaler_{timestamp}.pkl"
        encoder_file = models_dir / f"century_predictor_encoder_{timestamp}.pkl"
        
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
        logger.info("Testing century prediction with your palette...")
        logger.info("Palette colors:")
        for i, (color, weight) in enumerate(test_palette):
            logger.info(f"  {i+1}. {color} (weight: {weight:.2f})")
        
        # Make prediction
        predicted_century, confidence = predict_century_from_raw_palette(
            test_palette, model, scaler, encoder
        )
        
        logger.info("=" * 50)
        logger.info("CENTURY PREDICTION RESULT:")
        logger.info(f"Predicted Century: {predicted_century}")
        logger.info(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        logger.info("=" * 50)
        
        # Show what the vectorized palette looks like
        vectorized = vectorize_palette(test_palette)
        logger.info(f"Vectorized palette (first 10 values): {vectorized[:10]}")
        logger.info(f"Vectorized palette length: {len(vectorized)}")
        
        # Show available centuries
        logger.info(f"Available centuries in model: {sorted(encoder.classes_)}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
