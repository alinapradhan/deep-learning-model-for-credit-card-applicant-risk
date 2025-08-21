"""
Train the credit card fraud detection model.
"""
import os
import sys
import numpy as np
import pandas as pd
from data_preprocessing import prepare_data
from model import create_and_train_model
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main training pipeline."""
    logger.info("Starting credit card fraud detection model training...")
    
    try:
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Prepare data
        logger.info("Preparing data...")
        X_train, X_test, y_train, y_test, preprocessor = prepare_data()
        
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Test data shape: {X_test.shape}")
        logger.info(f"Class distribution in training set:")
        logger.info(pd.Series(y_train).value_counts().sort_index())
        
        # Train model
        logger.info("Training model...")
        model, results = create_and_train_model(
            X_train, y_train, X_test, y_test,
            validation_split=0.1,
            epochs=50  # Reduced for faster training
        )
        
        # Display results
        logger.info(f"Training completed successfully!")
        logger.info(f"Final Test AUC Score: {results['auc_score']:.4f}")
        
        # Print classification report
        from sklearn.metrics import classification_report
        logger.info("Classification Report:")
        logger.info("\n" + classification_report(y_test, results['predictions']))
        
        # Create visualization plots
        try:
            model.plot_training_history()
            model.plot_confusion_matrix(results['confusion_matrix'])
            model.plot_roc_curve(y_test, results['probabilities'])
            logger.info("Plots saved to models/ directory")
        except Exception as e:
            logger.warning(f"Could not create plots: {e}")
        
        logger.info("Model training and saving completed successfully!")
        logger.info("You can now run the API server with: python api.py")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()