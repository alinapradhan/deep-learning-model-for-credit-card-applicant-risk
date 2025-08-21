"""
Deep learning model for credit card fraud detection.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

class CreditRiskModel:
    def __init__(self, input_dim=None):
        self.model = None
        self.input_dim = input_dim
        self.history = None
        
    def build_model(self, input_dim):
        """Build the deep learning model architecture."""
        self.input_dim = input_dim
        
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile with appropriate metrics for binary classification
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'auc']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=100, batch_size=32, verbose=1):
        """Train the model."""
        if self.model is None:
            self.build_model(X_train.shape[1])
        
        # Callbacks for better training
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Prepare validation data
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.history
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        probabilities = self.model.predict(X)
        return probabilities.flatten()
    
    def predict_proba(self, X):
        """Return prediction probabilities."""
        return self.predict(X)
    
    def predict_classes(self, X, threshold=0.5):
        """Predict binary classes based on threshold."""
        probabilities = self.predict(X)
        return (probabilities >= threshold).astype(int)
    
    def evaluate(self, X_test, y_test, threshold=0.5):
        """Comprehensive model evaluation."""
        # Get predictions
        y_pred_proba = self.predict(X_test)
        y_pred = self.predict_classes(X_test, threshold)
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'auc_score': auc_score,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def plot_training_history(self):
        """Plot training history."""
        if self.history is None:
            print("No training history available.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Accuracy
        axes[0, 1].plot(self.history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in self.history.history:
            axes[0, 1].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        
        # AUC
        axes[1, 0].plot(self.history.history['auc'], label='Training AUC')
        if 'val_auc' in self.history.history:
            axes[1, 0].plot(self.history.history['val_auc'], label='Validation AUC')
        axes[1, 0].set_title('Model AUC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        
        # Precision
        axes[1, 1].plot(self.history.history['precision'], label='Training Precision')
        if 'val_precision' in self.history.history:
            axes[1, 1].plot(self.history.history['val_precision'], label='Validation Precision')
        axes[1, 1].set_title('Model Precision')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, cm, class_names=['No Default', 'Default']):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, y_test, y_pred_proba):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig('models/roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, file_path='models/credit_risk_model.h5'):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save model
        self.model.save(file_path)
        
        # Save input dimension
        joblib.dump(self.input_dim, file_path.replace('.h5', '_input_dim.pkl'))
        
        print(f"Model saved to {file_path}")
    
    def load_model(self, file_path='models/credit_risk_model.h5'):
        """Load a trained model."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        self.model = keras.models.load_model(file_path)
        
        # Load input dimension
        input_dim_file = file_path.replace('.h5', '_input_dim.pkl')
        if os.path.exists(input_dim_file):
            self.input_dim = joblib.load(input_dim_file)
        
        print(f"Model loaded from {file_path}")
    
    def get_feature_importance_scores(self, X_sample, feature_names):
        """Get simple feature importance using gradient-based method."""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        # Simple gradient-based importance
        X_tensor = tf.constant(X_sample.astype(np.float32))
        
        with tf.GradientTape() as tape:
            tape.watch(X_tensor)
            predictions = self.model(X_tensor)
        
        gradients = tape.gradient(predictions, X_tensor)
        importance_scores = tf.reduce_mean(tf.abs(gradients), axis=0).numpy()
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        return importance_df

def create_and_train_model(X_train, y_train, X_test, y_test, 
                          validation_split=0.1, epochs=100):
    """Create and train a complete model."""
    
    # Create validation split
    val_size = int(len(X_train) * validation_split)
    indices = np.random.permutation(len(X_train))
    train_idx, val_idx = indices[val_size:], indices[:val_size]
    
    X_train_split = X_train.iloc[train_idx] if isinstance(X_train, pd.DataFrame) else X_train[train_idx]
    y_train_split = y_train.iloc[train_idx] if isinstance(y_train, pd.Series) else y_train[train_idx]
    X_val_split = X_train.iloc[val_idx] if isinstance(X_train, pd.DataFrame) else X_train[val_idx]
    y_val_split = y_train.iloc[val_idx] if isinstance(y_train, pd.Series) else y_train[val_idx]
    
    # Create and train model
    model = CreditRiskModel()
    model.build_model(X_train.shape[1])
    
    print("Training model...")
    model.train(X_train_split, y_train_split, X_val_split, y_val_split, epochs=epochs)
    
    # Evaluate model
    print("\nEvaluating model...")
    results = model.evaluate(X_test, y_test)
    
    print(f"Test AUC Score: {results['auc_score']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, results['predictions']))
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model.save_model()
    
    return model, results

if __name__ == "__main__":
    from data_preprocessing import prepare_data
    
    # Prepare data
    print("Preparing data...")
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()
    
    # Train model
    model, results = create_and_train_model(X_train, y_train, X_test, y_test)
    
    # Plot results
    model.plot_training_history()
    model.plot_confusion_matrix(results['confusion_matrix'])
    model.plot_roc_curve(y_test, results['probabilities'])