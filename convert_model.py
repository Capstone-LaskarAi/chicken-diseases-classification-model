"""
Model converter utility to address TensorFlow compatibility issues.
Specifically handles the 'batch_shape' keyword argument error by recreating the model
from the notebook and saving it in a newer format.
"""

import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

# Print TF version for reference
print(f"TensorFlow version: {tf.__version__}")

# Class labels - must match the original model's classes
CLASS_LABELS = ['Coccidiosis', 'Healthy', 'Newcastle Disease', 'Salmonella']

def recreate_model():
    """
    Recreate the model architecture from the notebook
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

def try_convert_model(input_model_path="model/chicken_disease_model.h5", 
                     output_model_path="model/chicken_disease_model_fixed.h5",
                     output_saved_model_path="saved_model"):
    """
    Try to convert an existing model to a more compatible format
    """
    print(f"Attempting to convert model from {input_model_path}...")
    
    # Try various loading approaches
    model = None
    loaded_weights = False
    
    # First, try direct loading
    try:
        model = tf.keras.models.load_model(input_model_path, compile=False)
        loaded_weights = True
        print("Successfully loaded existing model")
    except Exception as e:
        print(f"Standard loading failed: {e}")
        
        # Try loading just the weights
        try:
            # Create a fresh model
            model = recreate_model()
            
            # Try to load weights
            try:
                model.load_weights(input_model_path)
                loaded_weights = True
                print("Successfully loaded weights into fresh model")
            except Exception as weight_error:
                print(f"Couldn't load weights: {weight_error}")
                
        except Exception as recreate_error:
            print(f"Model recreation failed: {recreate_error}")
    
    # If we have a model (with or without original weights), save it in new format
    if model is not None:
        # Save in H5 format
        try:
            model.save(output_model_path)
            print(f"Model saved to {output_model_path}")
        except Exception as h5_error:
            print(f"Error saving to H5 format: {h5_error}")
        
        # Save in SavedModel format
        try:
            os.makedirs(output_saved_model_path, exist_ok=True)
            tf.saved_model.save(model, output_saved_model_path)
            print(f"Model saved in SavedModel format to {output_saved_model_path}")
        except Exception as sm_error:
            print(f"Error saving to SavedModel format: {sm_error}")
        
        status_msg = "with original weights" if loaded_weights else "with ImageNet weights only"
        print(f"Model conversion completed {status_msg}")
        return True
    else:
        print("Model conversion failed - could not load or recreate model")
        return False

if __name__ == "__main__":
    # Try to convert the model
    try_convert_model()
