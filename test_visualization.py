"""
Quick test for visualization module setup.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.utils.visualization import PairsVisualization, ModelValidation
    print("✅ Visualization module imported successfully")
    
    # Create sample data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    
    data1 = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(len(dates)) * 0.02)
    }, index=dates)
    
    data2 = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(len(dates)) * 0.02)
    }, index=dates)
    
    # Test visualization
    viz = PairsVisualization()
    
    # Test correlation matrix
    corr_data = pd.DataFrame({
        'AAPL': np.random.randn(100),
        'MSFT': np.random.randn(100),
        'GOOGL': np.random.randn(100)
    }).corr()
    
    fig = viz.plot_correlation_heatmap(corr_data, show=False)
    print("✅ Correlation heatmap created successfully")
    
    # Test model validation
    residuals = pd.Series(np.random.randn(100))
    fitted = pd.Series(np.random.randn(100))
    
    jb_test = ModelValidation.jarque_bera_test(residuals)
    adf_test = ModelValidation.adf_stationarity_test(residuals)
    
    print("✅ Model validation tests working")
    print(f"JB test p-value: {jb_test['pvalue']:.4f}")
    print(f"ADF test p-value: {adf_test['pvalue']:.4f}")
    
    print("✅ All visualization tests passed!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
