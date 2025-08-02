import os
import sys
import numpy as np

# Add the parent directory to path to import the main module
sys.path.append(os.path.dirname(__file__))

try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    print("TensorFlow not found. Please install it using: pip install tensorflow")
    sys.exit(1)

try:
    from uncertainty_aware_rul_prediction import RULPredictor, print_metrics
    print("Successfully imported RULPredictor")
except ImportError as e:
    print(f"Error importing RULPredictor: {e}")
    sys.exit(1)

def test_with_small_dataset():
    """Test the implementation with a small subset of data."""
    print("Testing Uncertainty-Aware RUL Prediction with small dataset...")
    
    # Define file paths
    base_path = os.path.join(os.path.dirname(__file__), 'data', 'raw', 'CMaps')
    train_file = os.path.join(base_path, 'train_FD001.txt')
    test_file = os.path.join(base_path, 'test_FD001.txt')
    rul_file = os.path.join(base_path, 'RUL_FD001.txt')
    
    # Check if files exist
    for file_path in [train_file, test_file, rul_file]:
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            print("Please ensure the NASA C-MAPSS dataset files are in the correct location.")
            return False
    
    try:
        # Initialize predictor with smaller parameters for testing
        predictor = RULPredictor(
            sequence_length=10,  # Smaller sequence length
            n_features=20,       # Will be adjusted based on data
            n_models=2,          # Fewer models for faster training
            mc_samples=10        # Fewer MC samples
        )
        
        print("Loading and preprocessing data...")
        # Preprocess data
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = predictor.preprocess_data(
            train_file, test_file, rul_file
        )
        
        # Update n_features based on actual data
        predictor.n_features = X_train.shape[-1]
        print(f"Using {predictor.n_features} features")
        
        # Use only a small subset for testing
        subset_size = min(1000, len(X_train))
        X_train_small = X_train[:subset_size]
        y_train_small = y_train[:subset_size]
        
        subset_val_size = min(200, len(X_val))
        X_val_small = X_val[:subset_val_size]
        y_val_small = y_val[:subset_val_size]
        
        subset_test_size = min(200, len(X_test))
        X_test_small = X_test[:subset_test_size]
        y_test_small = y_test[:subset_test_size]
        
        print(f"Training on {len(X_train_small)} samples")
        print(f"Validating on {len(X_val_small)} samples")
        print(f"Testing on {len(X_test_small)} samples")
        
        # Train ensemble with fewer epochs
        print("Training ensemble...")
        predictor.train_ensemble(
            X_train_small, y_train_small, 
            X_val_small, y_val_small, 
            epochs=5,  # Very few epochs for testing
            batch_size=32
        )
        
        # Quick calibration test
        print("Testing calibration...")
        predictor.temperature_scaling(X_val_small, y_val_small)
        
        # Evaluate on test subset
        print("Evaluating model...")
        metrics, results = predictor.evaluate(X_test_small, y_test_small)
        
        # Print results
        print_metrics(metrics)
        
        # Test saving/loading
        print("Testing model saving...")
        test_save_dir = os.path.join(os.path.dirname(__file__), 'test_saved_models')
        predictor.save_models(test_save_dir)
        
        # Test loading
        print("Testing model loading...")
        new_predictor = RULPredictor()
        new_predictor.load_models(test_save_dir)
        
        print("✓ All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up test files
        test_save_dir = os.path.join(os.path.dirname(__file__), 'test_saved_models')
        if os.path.exists(test_save_dir):
            import shutil
            shutil.rmtree(test_save_dir)
            print("Cleaned up test files.")

if __name__ == "__main__":
    success = test_with_small_dataset()
    if success:
        print("\n" + "="*50)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("The full implementation is ready to run.")
        print("To run the complete training, execute:")
        print("python uncertainty_aware_rul_prediction.py")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("TEST FAILED!")
        print("Please check the error messages above.")
        print("="*50)
