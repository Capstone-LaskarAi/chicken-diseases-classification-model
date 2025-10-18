import os
import tensorflow as tf
import numpy as np
import streamlit as st
import traceback
from tensorflow.keras.preprocessing import image

# Konfigurasi
IMG_SIZE = (224, 224)
CLASS_LABELS = ['Coccidiosis', 'Healthy', 'Newcastle Disease', 'Salmonella']

# Model cache to avoid repeated loading
_MODEL_CACHE = {"model": None}

# Check for batch_shape error fix availability
try:
    from fix_batch_shape_error import fix_batch_shape_error, create_fresh_model
    batch_shape_fix_available = True
    print("Batch shape error fix available")
except ImportError:
    batch_shape_fix_available = False
    print("Batch shape error fix not available")

# Try to import model utils for direct inference
try:
    from model_utils_fallback import rebuild_model_from_scratch, direct_inference
    model_utils_fallback_available = True
    print("Model utilities fallback imported successfully")
except ImportError:
    model_utils_fallback_available = False
    print("Model utilities fallback not available - will use primary loading method only")

# Preprocess the image
def preprocess_image(img):
    """Preprocess image for model input"""
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Model conversion utility (for compatibility issues)
def convert_model_if_needed(model_path="model/fix_chicken_disease_model.h5", force_convert=False):
    """
    Convert the model to a format compatible with the current TF version if needed.
    This is useful when there are version compatibility issues.
    """
    
    converted_model_path = "chicken_disease_model_converted.h5"
    
    # Skip if converted model exists and force_convert is False
    if os.path.exists(converted_model_path) and not force_convert:
        try:
            model = tf.keras.models.load_model(converted_model_path)
            st.success("Loaded previously converted model")
            return model
        except Exception:
            # If loading the converted model fails, continue with conversion
            pass
    
    try:
        # Try to load the original model
        try:
            original_model = tf.keras.models.load_model(model_path, compile=False)
        except Exception as e:
            # If loading fails with standard method, try with tf.saved_model.load
            if "Unrecognized keyword arguments: ['batch_shape']" in str(e):
                st.warning("Attempting to load model with SavedModel API...")
                # This is a different loading mechanism that might work for older models
                original_model = tf.saved_model.load(model_path)
            else:
                raise e
        
        # Clone the model architecture
        new_model = tf.keras.models.clone_model(original_model)
        
        # Copy weights (this avoids serialization issues with custom layers)
        new_model.set_weights(original_model.get_weights())
        
        # Compile the model
        new_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        # Save the converted model
        new_model.save(converted_model_path)
        st.success(f"Model successfully converted and saved as {converted_model_path}")
        return new_model
    
    except Exception as e:
        st.error(f"Error converting model: {str(e)}")
        return None

# Load the model with special handling for TensorFlow compatibility issues
@st.cache_resource
def load_model():
    """
    Cached loader to avoid repeated disk I/O and initialization costs.
    """
    # Check if model is already cached
    if _MODEL_CACHE["model"] is not None:
        return _MODEL_CACHE["model"]
    
    try:
        # First try the fixed model if it exists
        if os.path.exists("model/updated_model.h5"):
            try:
                st.info("Attempting to load fixed model version...")
                model = tf.keras.models.load_model("model/updated_model.h5")
                st.success("Fixed model loaded successfully!")
                _MODEL_CACHE["model"] = model
                return model
            except Exception as fixed_error:
                st.warning(f"Fixed model loading failed: {str(fixed_error)}")
        
        # Try loading from saved_model directory if it exists
        if os.path.exists("saved_model"):
            try:
                st.info("Attempting to load from saved_model directory...")
                # Use the saved model directory which should be more compatible
                model = tf.saved_model.load("saved_model")
                st.success("Model loaded from saved_model directory!")
                _MODEL_CACHE["model"] = model
                return model
            except Exception as saved_model_error:
                st.warning(f"SavedModel loading failed: {str(saved_model_error)}")
        
        # If we reach here, try with the original model file
        try:
            st.info("Attempting standard model loading...")
            model = tf.keras.models.load_model("model/fix_chicken_disease_model.h5")
            st.success("Model loaded successfully!")
            _MODEL_CACHE["model"] = model
            return model
        except Exception as first_error:
            error_msg = str(first_error)
            st.warning(f"Standard model loading failed: {error_msg}")
            
            # Check for the batch_shape error specifically
            if "Unrecognized keyword arguments: ['batch_shape']" in error_msg:
                st.info("Detected 'batch_shape' error - attempting to fix...")
                
                # Use dedicated batch_shape error fix if available
                if batch_shape_fix_available:
                    try:
                        st.warning("Running batch_shape error fix...")
                        success = fix_batch_shape_error()
                        
                        if success:
                            st.success("Batch shape error fixed successfully!")
                            # Attempt to load the fixed model
                            fixed_model_path = "model/updated_model.h5"
                            if os.path.exists(fixed_model_path):
                                try:
                                    fixed_model = tf.keras.models.load_model(fixed_model_path)
                                    st.success("Fixed model loaded successfully!")
                                    return fixed_model
                                except Exception as fixed_load_error:
                                    st.warning(f"Error loading fixed model: {str(fixed_load_error)}")
                    except Exception as fix_error:
                        st.error(f"Error applying batch_shape fix: {str(fix_error)}")
                
                # Fallback to rebuild model from scratch
                if model_utils_fallback_available:
                    try:
                        # Create a fresh model with the same architecture
                        new_model = rebuild_model_from_scratch()
                        
                        if new_model:
                            st.warning("Model rebuilt from scratch. Using ImageNet weights (not trained weights).")
                            return new_model
                    except Exception as rebuild_error:
                        st.error(f"Model rebuilding failed: {str(rebuild_error)}")
            
            # If not the batch_shape error or rebuild failed, try with compile=False
            try:
                st.info("Attempting to load with compile=False...")
                # Custom objects may help with some layers
                custom_objects = {}
                
                model = tf.keras.models.load_model(
                    "model/fix_chicken_disease_model.h5", 
                    compile=False,
                    custom_objects=custom_objects
                )
                
                # Recompile the model
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                    loss="categorical_crossentropy", 
                    metrics=["accuracy"]
                )
                
                st.success("Model loaded successfully in compatibility mode!")
                return model
            except Exception as second_error:
                st.warning(f"Loading with compile=False failed: {str(second_error)}")
                
                # Final fallback - use model_utils if available
                if model_utils_fallback_available:
                    try:
                        st.warning("Attempting direct inference using rebuilt model...")
                        # Just return None and let predict_disease use direct_inference
                        return None
                    except Exception as utils_error:
                        st.error(f"Model utils fallback failed: {str(utils_error)}")
                
                # If we get here, all attempts failed
                st.error("All model loading methods failed")
                st.error("This is likely a TensorFlow version compatibility issue")
                return None
                
    except Exception as e:
        st.error(f"Unexpected error in model loading: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

# Predict disease
def predict_disease(model, img_array):
    try:
        # First attempt: standard prediction with model
        if model is not None:
            try:
                # Use a smaller batch_size and disable verbose output to avoid potential issues
                prediction = model.predict(img_array, batch_size=1, verbose=0)
                predicted_class_idx = np.argmax(prediction)
                predicted_class = CLASS_LABELS[predicted_class_idx]
                confidence = float(prediction[0][predicted_class_idx])
                return predicted_class, confidence
            except Exception as predict_error:
                st.error(f"Error during prediction with loaded model: {str(predict_error)}")
                st.error(f"Detailed error: {traceback.format_exc()}")
                
                # Try using __call__ method instead of predict for SavedModel format
                try:
                    if hasattr(model, '__call__'):
                        st.warning("Attempting to use model.__call__ instead of predict...")
                        prediction = model(img_array, training=False).numpy()
                        predicted_class_idx = np.argmax(prediction)
                        predicted_class = CLASS_LABELS[predicted_class_idx]
                        confidence = float(prediction[0][predicted_class_idx])
                        return predicted_class, confidence
                except Exception as call_error:
                    st.error(f"Error using model.__call__: {str(call_error)}")
        
        # Fallback: Use direct_inference from model_utils if available
        if model_utils_fallback_available:
            st.warning("Attempting to use direct_inference with rebuilt model...")
            try:
                predicted_class, confidence = direct_inference(img_array)
                if predicted_class:
                    st.warning("Used rebuilt model for prediction (results may be less accurate)")
                    return predicted_class, confidence
            except Exception as direct_error:
                st.error(f"Error in direct_inference: {str(direct_error)}")
        
        st.error("All prediction methods failed")
        return None, None
    except Exception as e:
        st.error(f"Unexpected error in prediction: {str(e)}")
        st.error(f"Detailed error: {traceback.format_exc()}")
        return None, None