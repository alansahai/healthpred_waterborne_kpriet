"""
Model package for outbreak prediction
"""
from .train import OutbreakModelTrainer
from .predict import OutbreakPredictor

# Backward compatibility - unified class for simple POC app
HealthOutbreakPredictor = OutbreakModelTrainer

__all__ = ['OutbreakModelTrainer', 'OutbreakPredictor', 'HealthOutbreakPredictor']
