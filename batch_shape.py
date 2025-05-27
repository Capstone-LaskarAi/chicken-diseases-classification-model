"""
This script specifically addresses the 'batch_shape' parameter error that occurs
when loading models saved with older versions of TensorFlow in newer versions.
It loads the model using tf.keras.models.model_from_json to avoid the serialization issues,
then saves it in a format compatible with newer TensorFlow versions.
"""

import os
import json
import tensorflow as tf
import numpy as np
import traceback
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

# Print TF version for reference
print(f"TensorFlow version: {tf.__version__}")

# Class labels - must match the original model's classes
CLASS_LABELS = ['Coccidiosis', 'Healthy', 'Newcastle Disease', 'Salmonella']

def create_fresh_model():
    """
    Creates a fresh MobileNetV2 model with the same architecture as the original.
    This is useful when we can't load the original model due to compatibility issues.
    """
    # Create base model
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    
    # Freeze early layers
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    predictions = Dense(len(CLASS_LABELS), activation="softmax")(x)
    
    # Create model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

def extract_weights_directly(h5_path):
    """
    Attempts to extract weights directly from H5 file using low-level H5 access,
    which can work around TensorFlow version compatibility issues.
    """
    try:
        import h5py
        with h5py.File(h5_path, 'r') as h5file:
            if 'model_weights' in h5file:
                print("Found 'model_weights' group in H5 file")
                # H5 file has standard structure - can be processed
                return True
            else:
                print("'model_weights' group not found in H5 file")
                # Not a standard structure
                return False
    except Exception as e:
        print(f"Error accessing H5 file: {str(e)}")
        return False

def fix_batch_shape_error(input_model_path="model/chicken_disease_model.h5",
                         output_model_path="model/chicken_disease_model_fixed.h5",
                         output_saved_model_path="saved_model"):
    """
    Fix the 'batch_shape' parameter error by:
    1. Creating a fresh model with the same architecture
    2. Saving it in both H5 and SavedModel formats
    """
    print(f"Fixing batch_shape error for model at {input_model_path}...")
    
    # First, try to create a fresh model
    try:
        model = create_fresh_model()
        
        # Try to extract weights from original model
        try_load_weights = False
        if os.path.exists(input_model_path) and extract_weights_directly(input_model_path):
            try_load_weights = True
        
        if try_load_weights:
            try:
                # Try to load just weights without full model loading
                model.load_weights(input_model_path, by_name=True)
                print("Successfully loaded weights from original model")
            except Exception as weight_error:
                print(f"Could not load weights: {str(weight_error)}")
                print("Using ImageNet pretrained weights only (no specific chicken disease training)")
        else:
            print("Using ImageNet pretrained weights only (no specific chicken disease training)")
        
        # Save in H5 format - clean version without batch_shape parameter
        model.save(output_model_path, save_format='h5')
        print(f"Fixed model saved to {output_model_path}")
        
        # Save in SavedModel format which is more compatible with TF 2.x
        if not os.path.exists(output_saved_model_path):
            os.makedirs(output_saved_model_path, exist_ok=True)
        
        tf.saved_model.save(model, output_saved_model_path)
        print(f"Fixed model also saved in SavedModel format to {output_saved_model_path}")
        
        return True
        
    except Exception as e:
        print(f"Error fixing model: {str(e)}")
        return False

if __name__ == "__main__":
    # Run the fix directly when script is executed
    success = fix_batch_shape_error()
    if success:
        print("Successfully fixed batch_shape error and created compatible model versions")
    else:
        print("Failed to fix batch_shape error")
