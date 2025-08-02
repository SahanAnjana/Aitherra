import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class RULPredictor:
    
    def __init__(self, sequence_length=30, n_features=24, n_models=5, mc_samples=50):
        """        
        Args:
            sequence_length (int): Length of input sequences
            n_features (int): Number of sensor features (excluding unit, cycle, operational settings)
            n_models (int): Number of models in the ensemble
            mc_samples (int): Number of Monte Carlo samples for uncertainty estimation
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_models = n_models
        self.mc_samples = mc_samples
        self.models = []
        self.scalers = {'features': MinMaxScaler(), 'targets': MinMaxScaler()}
        self.temperature = 1.0
        self.abstention_threshold = 50.0  # RUL cycles
        
    def load_data(self, train_file, test_file, rul_file):
        """        
        Args:
            train_file (str): Path to training data file
            test_file (str): Path to test data file
            rul_file (str): Path to RUL labels file
            
        Returns:
            tuple: Processed training and test datasets
        """
        print("Loading and preprocessing data...")
        
        # Define column names
        columns = ['unit', 'cycle'] + [f'setting_{i}' for i in range(1, 4)] + \
                 [f'sensor_{i}' for i in range(1, 22)]
          # Load data
        train_df = pd.read_csv(train_file, sep=r'\s+', header=None, names=columns)
        test_df = pd.read_csv(test_file, sep=r'\s+', header=None, names=columns)
        rul_df = pd.read_csv(rul_file, sep=r'\s+', header=None, names=['RUL'])
        
        # Calculate RUL for training data
        train_df = self._calculate_rul(train_df)
        
        # Add RUL to test data
        test_df = self._add_test_rul(test_df, rul_df)
        
        # Remove constant sensors (no variation)
        constant_sensors = []
        for col in train_df.columns:
            if col.startswith('sensor_') and train_df[col].nunique() <= 1:
                constant_sensors.append(col)
        
        if constant_sensors:
            print(f"Removing constant sensors: {constant_sensors}")
            train_df = train_df.drop(columns=constant_sensors)
            test_df = test_df.drop(columns=constant_sensors)
        
        # Select features (operational settings + sensors)
        feature_cols = [col for col in train_df.columns 
                       if col.startswith('setting_') or col.startswith('sensor_')]
        
        print(f"Using {len(feature_cols)} features: {feature_cols}")
        
        return train_df, test_df, feature_cols
    
    def _calculate_rul(self, df):
        df = df.copy()
        df['RUL'] = 0
        
        for unit in df['unit'].unique():
            unit_data = df[df['unit'] == unit]
            max_cycle = unit_data['cycle'].max()
            df.loc[df['unit'] == unit, 'RUL'] = max_cycle - df.loc[df['unit'] == unit, 'cycle']
        
        return df
    def _add_test_rul(self, test_df, rul_df):
        """Add RUL labels to test data."""
        test_df = test_df.copy()
        test_df['RUL'] = 0
        
        for i, unit in enumerate(sorted(test_df['unit'].unique())):
            unit_data = test_df[test_df['unit'] == unit]
            max_cycle = unit_data['cycle'].max()
            true_rul = rul_df.iloc[i]['RUL']
            
            # RUL for each cycle in the test unit
            test_df.loc[test_df['unit'] == unit, 'RUL'] = true_rul + (max_cycle - test_df.loc[test_df['unit'] == unit, 'cycle'])
        
        return test_df
    
    def create_sequences(self, df, feature_cols):
        """        
        Args:
            df (DataFrame): Input dataframe
            feature_cols (list): List of feature column names
            
        Returns:
            tuple: (X, y) sequences and targets
        """
        print(f"Creating sequences with window size {self.sequence_length}...")
        
        X, y = [], []
        
        for unit in df['unit'].unique():
            unit_data = df[df['unit'] == unit].sort_values('cycle')
            unit_features = unit_data[feature_cols].values
            unit_targets = unit_data['RUL'].values
            
            # Create sequences
            for i in range(len(unit_data) - self.sequence_length + 1):
                X.append(unit_features[i:i + self.sequence_length])
                y.append(unit_targets[i + self.sequence_length - 1])
        
        return np.array(X), np.array(y)
    
    def preprocess_data(self, train_file, test_file, rul_file):
        """        
        Args:
            train_file (str): Path to training data file
            test_file (str): Path to test data file
            rul_file (str): Path to RUL labels file
            
        Returns:
            tuple: Processed data splits
        """
        # Load data
        train_df, test_df, feature_cols = self.load_data(train_file, test_file, rul_file)
        
        # Create sequences
        X_train_full, y_train_full = self.create_sequences(train_df, feature_cols)
        X_test, y_test = self.create_sequences(test_df, feature_cols)
        
        # Split training data into train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=42
        )
        
        # Normalize features
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        self.scalers['features'].fit(X_train_reshaped)
        
        X_train = self._normalize_sequences(X_train)
        X_val = self._normalize_sequences(X_val)
        X_test = self._normalize_sequences(X_test)
        
        # Normalize targets
        self.scalers['targets'].fit(y_train.reshape(-1, 1))
        y_train = self.scalers['targets'].transform(y_train.reshape(-1, 1)).flatten()
        y_val = self.scalers['targets'].transform(y_val.reshape(-1, 1)).flatten()
        y_test = self.scalers['targets'].transform(y_test.reshape(-1, 1)).flatten()
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")
        print(f"Test data shape: {X_test.shape}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def _normalize_sequences(self, X):
        X_norm = np.zeros_like(X)
        for i in range(X.shape[0]):
            X_norm[i] = self.scalers['features'].transform(X[i])
        return X_norm
    
    def create_model(self, dropout_rate=0.3, seed=None):
        """        
        Args:
            dropout_rate (float): Dropout rate
            seed (int): Random seed for initialization
            
        Returns:
            tf.keras.Model: Compiled model
        """
        if seed is not None:
            tf.random.set_seed(seed)
        
        model = keras.Sequential([
            layers.LSTM(64, return_sequences=True, input_shape=(self.sequence_length, self.n_features)),
            layers.Dropout(dropout_rate),
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(dropout_rate),
            layers.Dense(128, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(64, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(32, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(1, activation='linear')
        ])
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_ensemble(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=64):
        """        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs (int): Number of training epochs
            batch_size (int): Batch size
        """
        print(f"Training ensemble of {self.n_models} models...")
        
        self.models = []
        
        for i in range(self.n_models):
            print(f"Training model {i+1}/{self.n_models}")
            
            # Create model with different seed
            model = self.create_model(seed=42 + i)
            
            # Callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=20, restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7
                )
            ]
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0
            )
            
            self.models.append(model)
            
            # Print training results
            val_loss = min(history.history['val_loss'])
            print(f"Model {i+1} - Best validation loss: {val_loss:.4f}")
    
    def mc_dropout_predict(self, X, model_idx=0):
        """        
        Args:
            X (np.array): Input data
            model_idx (int): Index of model to use
            
        Returns:
            tuple: (mean_predictions, uncertainty_estimates)
        """
        model = self.models[model_idx]
        
        # Enable dropout during inference
        predictions = []
        for _ in range(self.mc_samples):
            # Set training=True to enable dropout
            pred = model(X, training=True)
            predictions.append(pred.numpy())
        
        predictions = np.array(predictions)
        
        # Calculate mean and variance
        mean_pred = np.mean(predictions, axis=0)
        var_pred = np.var(predictions, axis=0)
        
        return mean_pred.flatten(), var_pred.flatten()
    
    def ensemble_predict(self, X):
        """        
        Args:
            X (np.array): Input data
            
        Returns:
            tuple: (ensemble_mean, epistemic_uncertainty, aleatoric_uncertainty)
        """
        print("Generating ensemble predictions...")
        
        # Get predictions from each model (with MC Dropout)
        model_means = []
        model_vars = []
        
        for i in range(self.n_models):
            mean_pred, var_pred = self.mc_dropout_predict(X, i)
            model_means.append(mean_pred)
            model_vars.append(var_pred)
        
        model_means = np.array(model_means)
        model_vars = np.array(model_vars)
        
        # Ensemble mean
        ensemble_mean = np.mean(model_means, axis=0)
        
        # Epistemic uncertainty (variance between models)
        epistemic_uncertainty = np.var(model_means, axis=0)
        
        # Aleatoric uncertainty (average within-model variance)
        aleatoric_uncertainty = np.mean(model_vars, axis=0)
        
        return ensemble_mean, epistemic_uncertainty, aleatoric_uncertainty
    
    def temperature_scaling(self, X_val, y_val):
        """        
        Args:
            X_val (np.array): Validation input data
            y_val (np.array): Validation target data
        """
        print("Performing temperature scaling calibration...")
        
        # Get validation predictions
        val_mean, _, _ = self.ensemble_predict(X_val)
        
        # Convert to classification bins for calibration
        # Create bins based on RUL ranges
        n_bins = 10
        y_val_denorm = self.scalers['targets'].inverse_transform(y_val.reshape(-1, 1)).flatten()
        val_mean_denorm = self.scalers['targets'].inverse_transform(val_mean.reshape(-1, 1)).flatten()
        
        # Create bins
        max_rul = max(y_val_denorm.max(), val_mean_denorm.max())
        bin_edges = np.linspace(0, max_rul, n_bins + 1)
        
        # Convert to bin indices
        y_val_bins = np.digitize(y_val_denorm, bin_edges) - 1
        y_val_bins = np.clip(y_val_bins, 0, n_bins - 1)
        
        pred_bins = np.digitize(val_mean_denorm, bin_edges) - 1
        pred_bins = np.clip(pred_bins, 0, n_bins - 1)
        
        # Optimize temperature (simplified - using grid search)
        temperatures = np.logspace(-2, 2, 50)
        best_ece = float('inf')
        best_temp = 1.0
        
        for temp in temperatures:
            # Apply temperature scaling (simplified for regression)
            scaled_confidence = np.exp(-np.abs(val_mean_denorm - y_val_denorm) / temp)
            
            # Calculate ECE (simplified)
            ece = self._calculate_ece(y_val_bins, pred_bins, scaled_confidence, n_bins)
            
            if ece < best_ece:
                best_ece = ece
                best_temp = temp
        
        self.temperature = best_temp
        print(f"Optimal temperature: {self.temperature:.4f}")
        print(f"Expected Calibration Error: {best_ece:.4f}")
    
    def _calculate_ece(self, y_true, y_pred, confidence, n_bins):
        ece = 0.0
        n_samples = len(y_true)
        
        for i in range(n_bins):
            bin_mask = (y_pred == i)
            if np.sum(bin_mask) > 0:
                bin_accuracy = np.mean(y_true[bin_mask] == y_pred[bin_mask])
                bin_confidence = np.mean(confidence[bin_mask])
                bin_size = np.sum(bin_mask)
                
                ece += (bin_size / n_samples) * np.abs(bin_accuracy - bin_confidence)
        
        return ece
    
    def predict_with_abstention(self, X):
        """        
        Args:
            X (np.array): Input data
            
        Returns:
            dict: Predictions, uncertainties, and abstention flags
        """
        print("Making predictions with abstention mechanism...")
        
        # Get ensemble predictions
        mean_pred, epistemic_unc, aleatoric_unc = self.ensemble_predict(X)
        
        # Total uncertainty
        total_uncertainty = epistemic_unc + aleatoric_unc
        
        # Denormalize predictions
        mean_pred_denorm = self.scalers['targets'].inverse_transform(mean_pred.reshape(-1, 1)).flatten()
          # Calculate confidence intervals (±2σ)
        std_pred = np.sqrt(total_uncertainty)
        std_pred_denorm = std_pred * self.scalers['targets'].scale_[0]
        
        # Ensure minimum uncertainty for realistic confidence intervals
        min_std = 5.0  # Minimum 5 cycles uncertainty
        std_pred_denorm = np.maximum(std_pred_denorm, min_std)
        
        ci_lower = mean_pred_denorm - 2 * std_pred_denorm
        ci_upper = mean_pred_denorm + 2 * std_pred_denorm
        ci_width = ci_upper - ci_lower
        
        # Abstention decision
        abstain_mask = ci_width > self.abstention_threshold
        
        results = {
            'predictions': mean_pred_denorm,
            'epistemic_uncertainty': epistemic_unc,
            'aleatoric_uncertainty': aleatoric_unc,
            'total_uncertainty': total_uncertainty,
            'confidence_intervals': (ci_lower, ci_upper),
            'abstention_mask': abstain_mask,
            'abstention_rate': np.mean(abstain_mask)
        }
        
        return results
    
    def evaluate(self, X_test, y_test):
        """        
        Args:
            X_test (np.array): Test input data
            y_test (np.array): Test target data
            
        Returns:
            dict: Evaluation metrics
        """
        print("Evaluating model performance...")
        
        # Get predictions
        results = self.predict_with_abstention(X_test)
        
        # Denormalize true values
        y_test_denorm = self.scalers['targets'].inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_test_denorm, results['predictions']))
        
        # Calculate RMSE for non-abstained predictions
        non_abstain_mask = ~results['abstention_mask']
        if np.any(non_abstain_mask):
            rmse_non_abstain = np.sqrt(mean_squared_error(
                y_test_denorm[non_abstain_mask], 
                results['predictions'][non_abstain_mask]
            ))
        else:
            rmse_non_abstain = np.nan
        
        # Calculate coverage (95% confidence interval)
        ci_lower, ci_upper = results['confidence_intervals']
        in_interval = (y_test_denorm >= ci_lower) & (y_test_denorm <= ci_upper)
        coverage = np.mean(in_interval)
        
        # Calculate coverage for non-abstained predictions
        if np.any(non_abstain_mask):
            coverage_non_abstain = np.mean(in_interval[non_abstain_mask])
        else:
            coverage_non_abstain = np.nan
        
        metrics = {
            'rmse': rmse,
            'rmse_non_abstain': rmse_non_abstain,
            'coverage': coverage,
            'coverage_non_abstain': coverage_non_abstain,
            'abstention_rate': results['abstention_rate'],
            'mean_epistemic_uncertainty': np.mean(results['epistemic_uncertainty']),
            'mean_aleatoric_uncertainty': np.mean(results['aleatoric_uncertainty'])
        }
        
        return metrics, results
    def save_models(self, save_dir):
        """Save trained models and scalers."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save models using the new Keras format
        for i, model in enumerate(self.models):
            model.save(os.path.join(save_dir, f'model_{i}.keras'))
        
        # Save scalers
        with open(os.path.join(save_dir, 'scalers.pkl'), 'wb') as f:
            pickle.dump(self.scalers, f)
        
        # Save configuration
        config = {
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'n_models': self.n_models,
            'mc_samples': self.mc_samples,
            'temperature': self.temperature,
            'abstention_threshold': self.abstention_threshold
        }
        
        with open(os.path.join(save_dir, 'config.pkl'), 'wb') as f:
            pickle.dump(config, f)
        
        print(f"Models saved to {save_dir}")
    
    def load_models(self, save_dir):
        # Load configuration
        with open(os.path.join(save_dir, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
        
        self.sequence_length = config['sequence_length']
        self.n_features = config['n_features']
        self.n_models = config['n_models']
        self.mc_samples = config['mc_samples']
        self.temperature = config['temperature']
        self.abstention_threshold = config['abstention_threshold']
        
        # Load scalers
        with open(os.path.join(save_dir, 'scalers.pkl'), 'rb') as f:
            self.scalers = pickle.load(f)
        
        # Load models
        self.models = []
        for i in range(self.n_models):
            model_path = os.path.join(save_dir, f'model_{i}.keras')
            if not os.path.exists(model_path):
                # Try the old .h5 format for backward compatibility
                model_path = os.path.join(save_dir, f'model_{i}.h5')
            
            model = keras.models.load_model(model_path, compile=False)
            # Recompile the model to avoid serialization issues
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            self.models.append(model)
        
        print(f"Models loaded from {save_dir}")

def create_visualizations(predictor, X_test, y_test, results, metrics, save_dir):
    """    
    Args:
        predictor: Trained RUL predictor
        X_test: Test input data
        y_test: Test target data
        results: Prediction results
        metrics: Evaluation metrics
        save_dir: Directory to save plots
    """
    print("Creating visualizations...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Denormalize true values
    y_test_denorm = predictor.scalers['targets'].inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Predicted vs Actual RUL
    plt.figure(figsize=(12, 8))
    
    # All predictions
    plt.subplot(2, 2, 1)
    plt.scatter(y_test_denorm, results['predictions'], alpha=0.6, s=30)
    plt.plot([y_test_denorm.min(), y_test_denorm.max()], 
             [y_test_denorm.min(), y_test_denorm.max()], 'r--', lw=2)
    plt.xlabel('Actual RUL (cycles)')
    plt.ylabel('Predicted RUL (cycles)')
    plt.title(f'Predicted vs Actual RUL\nRMSE: {metrics["rmse"]:.2f}')
    plt.grid(True, alpha=0.3)
    
    # Non-abstained predictions
    plt.subplot(2, 2, 2)
    non_abstain = ~results['abstention_mask']
    if np.any(non_abstain):
        plt.scatter(y_test_denorm[non_abstain], results['predictions'][non_abstain], 
                   alpha=0.6, s=30, color='green', label='Non-abstained')
    
    abstain = results['abstention_mask']
    if np.any(abstain):
        plt.scatter(y_test_denorm[abstain], results['predictions'][abstain], 
                   alpha=0.6, s=30, color='red', label='Abstained')
    
    plt.plot([y_test_denorm.min(), y_test_denorm.max()], 
             [y_test_denorm.min(), y_test_denorm.max()], 'k--', lw=2)
    plt.xlabel('Actual RUL (cycles)')
    plt.ylabel('Predicted RUL (cycles)')
    plt.title(f'Predictions with Abstention\nAbstention Rate: {metrics["abstention_rate"]:.1%}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Uncertainty visualization
    plt.subplot(2, 2, 3)
    ci_lower, ci_upper = results['confidence_intervals']
    sample_indices = np.arange(min(len(y_test_denorm), 200))  # Show first 200 samples
    
    plt.fill_between(sample_indices, 
                     ci_lower[sample_indices], 
                     ci_upper[sample_indices], 
                     alpha=0.3, label='95% Confidence Interval')
    plt.plot(sample_indices, y_test_denorm[sample_indices], 'go', markersize=4, label='Actual RUL')
    plt.plot(sample_indices, results['predictions'][sample_indices], 'bo', markersize=4, label='Predicted RUL')
    
    plt.xlabel('Sample Index')
    plt.ylabel('RUL (cycles)')
    plt.title(f'Uncertainty Intervals\nCoverage: {metrics["coverage"]:.1%}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Uncertainty distribution
    plt.subplot(2, 2, 4)
    total_unc = np.sqrt(results['total_uncertainty'])
    total_unc_denorm = total_unc * predictor.scalers['targets'].scale_[0]
    
    plt.hist(total_unc_denorm, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(predictor.abstention_threshold/4, color='red', linestyle='--', 
                label=f'Abstention Threshold/4: {predictor.abstention_threshold/4:.1f}')
    plt.xlabel('Uncertainty (RUL cycles)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Uncertainty')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prediction_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Calibration plot (reliability diagram)
    plt.figure(figsize=(10, 6))
    
    # Calculate calibration
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    # Convert predictions to probabilities (simplified)
    pred_probs = 1 / (1 + np.abs(results['predictions'] - y_test_denorm) / 100)
    
    accuracies = []
    confidences = []
    bin_sizes = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (pred_probs > bin_lower) & (pred_probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.abs(results['predictions'][in_bin] - y_test_denorm[in_bin])
            accuracy_in_bin = np.mean(accuracy_in_bin < 20)  # Within 20 cycles
            avg_confidence_in_bin = pred_probs[in_bin].mean()
            
            accuracies.append(accuracy_in_bin)
            confidences.append(avg_confidence_in_bin)
            bin_sizes.append(prop_in_bin)
        else:
            accuracies.append(0)
            confidences.append(0)
            bin_sizes.append(0)
    
    plt.subplot(1, 2, 1)
    plt.bar(confidences, accuracies, width=0.08, alpha=0.7, edgecolor='black')
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.bar(confidences, bin_sizes, width=0.08, alpha=0.7, edgecolor='black')
    plt.xlabel('Confidence')
    plt.ylabel('Proportion of Samples')
    plt.title('Confidence Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'calibration_plot.png'), dpi=300, bbox_inches='tight')
    plt.show()

def print_metrics(metrics):
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    print(f"RMSE (All Predictions): {metrics['rmse']:.2f} cycles")
    print(f"RMSE (Non-Abstained): {metrics['rmse_non_abstain']:.2f} cycles")
    print(f"Coverage (95% CI): {metrics['coverage']:.1%}")
    print(f"Coverage (Non-Abstained): {metrics['coverage_non_abstain']:.1%}")
    print(f"Abstention Rate: {metrics['abstention_rate']:.1%}")
    print(f"Mean Epistemic Uncertainty: {metrics['mean_epistemic_uncertainty']:.4f}")
    print(f"Mean Aleatoric Uncertainty: {metrics['mean_aleatoric_uncertainty']:.4f}")
    print("="*50)

def main():
    print("Starting Uncertainty-Aware RUL Prediction...")
    
    # Define file paths (relative to the script location)
    base_path = os.path.join(os.path.dirname(__file__), 'data', 'raw', 'CMaps')
    train_file = os.path.join(base_path, 'train_FD001.txt')
    test_file = os.path.join(base_path, 'test_FD001.txt')
    rul_file = os.path.join(base_path, 'RUL_FD001.txt')
    
    # Check if files exist
    for file_path in [train_file, test_file, rul_file]:
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return
    
    # Initialize predictor
    predictor = RULPredictor(
        sequence_length=30,
        n_features=24,  # Will be adjusted based on available features
        n_models=5,
        mc_samples=50
    )
    
    try:
        # Preprocess data
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = predictor.preprocess_data(
            train_file, test_file, rul_file
        )
        
        # Update n_features based on actual data
        predictor.n_features = X_train.shape[-1]
        
        # Train ensemble
        predictor.train_ensemble(X_train, y_train, X_val, y_val, epochs=50, batch_size=64)
        
        # Calibrate with temperature scaling
        predictor.temperature_scaling(X_val, y_val)
        
        # Evaluate on test set
        metrics, results = predictor.evaluate(X_test, y_test)
        
        # Print metrics
        print_metrics(metrics)
        
        # Save models
        save_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
        predictor.save_models(save_dir)
        
        # Create visualizations
        viz_dir = os.path.join(os.path.dirname(__file__), 'visualizations')
        create_visualizations(predictor, X_test, y_test, results, metrics, viz_dir)
        
        print(f"\nComplete! Models saved to: {save_dir}")
        print(f"Visualizations saved to: {viz_dir}")
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
