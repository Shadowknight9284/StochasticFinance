"""
Cointegration testing module for pairs trading.
Implements various cointegration tests including Engle-Granger and Johansen tests.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.stattools import adfuller, coint
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    from statsmodels.regression.linear_model import OLS
    from statsmodels.stats.diagnostic import het_breuschpagan
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available. Install with: pip install statsmodels")

from scipy import stats
from sklearn.linear_model import LinearRegression


@dataclass
class CointegrationResult:
    """Results from cointegration testing."""
    is_cointegrated: bool
    test_statistic: float
    p_value: float
    critical_values: Dict[str, float]
    confidence_level: str
    hedge_ratio: float
    residuals: pd.Series
    half_life: Optional[float] = None
    
    def __post_init__(self):
        """Calculate additional metrics after initialization."""
        if self.residuals is not None and len(self.residuals) > 1:
            self.half_life = self._calculate_half_life()
    
    def _calculate_half_life(self) -> Optional[float]:
        """Calculate half-life of mean reversion."""
        try:
            # Fit AR(1) model to residuals
            y = self.residuals.dropna()
            if len(y) < 10:
                return None
                
            x = y.shift(1).dropna()
            y = y[1:]
            
            if len(x) != len(y) or len(x) == 0:
                return None
            
            # Simple linear regression
            reg = LinearRegression().fit(x.values.reshape(-1, 1), y.values)
            alpha = reg.coef_[0]
            
            if alpha >= 1 or alpha <= 0:
                return None
                
            half_life = -np.log(2) / np.log(alpha)
            return half_life if half_life > 0 else None
            
        except Exception:
            return None


@dataclass 
class JohansenResult:
    """Results from Johansen cointegration test."""
    trace_stat: float
    trace_critical_value: float
    max_eigen_stat: float
    max_eigen_critical_value: float
    is_cointegrated: bool
    eigenvectors: np.ndarray
    eigenvalues: np.ndarray


class CointegrationTester:
    """Comprehensive cointegration testing suite."""
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize cointegration tester.
        
        Args:
            significance_level: Statistical significance level for tests
        """
        self.significance_level = significance_level
        
        if not STATSMODELS_AVAILABLE:
            print("Warning: Some advanced tests unavailable without statsmodels")
    
    def engle_granger_test(self, y1: pd.Series, y2: pd.Series) -> CointegrationResult:
        """
        Perform Engle-Granger cointegration test.
        
        Args:
            y1, y2: Price series to test for cointegration
            
        Returns:
            CointegrationResult with test results
        """
        # Align series
        aligned_data = pd.concat([y1, y2], axis=1).dropna()
        if len(aligned_data) < 30:
            raise ValueError("Insufficient data for cointegration test")
        
        s1, s2 = aligned_data.iloc[:, 0], aligned_data.iloc[:, 1]
        
        # Step 1: Estimate hedge ratio using OLS
        if STATSMODELS_AVAILABLE:
            try:
                # Use statsmodels for better statistics
                model = OLS(s1, s2).fit()
                hedge_ratio = model.params.iloc[0]
                residuals = model.resid
            except Exception:
                # Fallback to sklearn
                hedge_ratio, residuals = self._simple_regression(s1, s2)
        else:
            hedge_ratio, residuals = self._simple_regression(s1, s2)
        
        # Step 2: Test residuals for stationarity
        if STATSMODELS_AVAILABLE:
            try:
                adf_stat, p_value, _, _, critical_values, _ = adfuller(
                    residuals, autolag='AIC', regression='c'
                )
                
                # Convert critical values to dict
                cv_dict = {
                    '1%': critical_values['1%'],
                    '5%': critical_values['5%'], 
                    '10%': critical_values['10%']
                }
            except Exception:
                adf_stat, p_value, cv_dict = self._simple_adf_test(residuals)
        else:
            adf_stat, p_value, cv_dict = self._simple_adf_test(residuals)
        
        # Determine cointegration
        critical_value = cv_dict[f'{int(self.significance_level * 100)}%']
        is_cointegrated = adf_stat < critical_value
        confidence_level = f"{int((1 - self.significance_level) * 100)}%"
        
        return CointegrationResult(
            is_cointegrated=is_cointegrated,
            test_statistic=adf_stat,
            p_value=p_value,
            critical_values=cv_dict,
            confidence_level=confidence_level,
            hedge_ratio=hedge_ratio,
            residuals=pd.Series(residuals, index=s1.index)
        )
    
    def _simple_regression(self, y: pd.Series, x: pd.Series) -> Tuple[float, np.ndarray]:
        """Simple linear regression fallback."""
        X = x.values.reshape(-1, 1)
        reg = LinearRegression().fit(X, y.values)
        hedge_ratio = reg.coef_[0]
        residuals = y.values - reg.predict(X)
        return hedge_ratio, residuals
    
    def _simple_adf_test(self, series: np.ndarray) -> Tuple[float, float, Dict[str, float]]:
        """Simplified ADF test using basic statistics."""
        # This is a very basic implementation - not as robust as statsmodels
        y = np.diff(series)
        x = series[:-1]
        
        if len(x) == 0 or np.std(x) == 0:
            return 0.0, 1.0, {'1%': -3.96, '5%': -3.41, '10%': -3.13}
        
        # Simple t-statistic
        corr = np.corrcoef(y, x)[0, 1] if len(y) > 1 else 0
        t_stat = corr * np.sqrt(len(y) - 2) / np.sqrt(1 - corr**2) if corr != 1 else 0
        
        # Approximate p-value
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(y) - 2)) if len(y) > 2 else 1.0
        
        # Standard critical values for ADF test
        critical_values = {'1%': -3.96, '5%': -3.41, '10%': -3.13}
        
        return t_stat, p_value, critical_values
    
    def johansen_test(self, data: pd.DataFrame) -> JohansenResult:
        """
        Perform Johansen cointegration test.
        
        Args:
            data: DataFrame with price series to test
            
        Returns:
            JohansenResult with test results
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("Johansen test requires statsmodels. Install with: pip install statsmodels")
        
        try:
            # Clean data
            clean_data = data.dropna()
            if len(clean_data) < 30:
                raise ValueError("Insufficient data for Johansen test")
            
            # Perform test
            result = coint_johansen(clean_data.values, det_order=0, k_ar_diff=1)
            
            # Extract results (test for at least 1 cointegrating relationship)
            trace_stat = result.lr1[0]  # Trace statistic for r=0
            trace_critical = result.cvt[0, 1]  # 5% critical value
            
            max_eigen_stat = result.lr2[0]  # Max eigenvalue statistic for r=0  
            max_eigen_critical = result.cvm[0, 1]  # 5% critical value
            
            # Check if cointegrated (either test)
            is_cointegrated = (trace_stat > trace_critical) or (max_eigen_stat > max_eigen_critical)
            
            return JohansenResult(
                trace_stat=trace_stat,
                trace_critical_value=trace_critical,
                max_eigen_stat=max_eigen_stat,
                max_eigen_critical_value=max_eigen_critical,
                is_cointegrated=is_cointegrated,
                eigenvectors=result.evec,
                eigenvalues=result.eig
            )
            
        except Exception as e:
            print(f"Johansen test failed: {e}")
            # Return default result
            return JohansenResult(
                trace_stat=0.0,
                trace_critical_value=20.26,
                max_eigen_stat=0.0, 
                max_eigen_critical_value=15.89,
                is_cointegrated=False,
                eigenvectors=np.array([]),
                eigenvalues=np.array([])
            )
    
    def test_pair_cointegration(self, price1: pd.Series, price2: pd.Series) -> Dict:
        """
        Comprehensive cointegration testing for a pair.
        
        Args:
            price1, price2: Price series for the pair
            
        Returns:
            Dictionary with comprehensive test results
        """
        try:
            # Engle-Granger test
            eg_result = self.engle_granger_test(price1, price2)
            
            # Johansen test (if available)
            data = pd.concat([price1, price2], axis=1).dropna()
            try:
                johansen_result = self.johansen_test(data)
            except Exception:
                johansen_result = None
            
            # Additional statistics
            correlation = price1.corr(price2)
            
            # Price ratio statistics
            ratio = price1 / price2
            ratio_mean = ratio.mean()
            ratio_std = ratio.std()
            
            # Log price spread
            log_spread = np.log(price1) - np.log(price2)
            spread_volatility = log_spread.std()
            
            return {
                'engle_granger': eg_result,
                'johansen': johansen_result,
                'correlation': correlation,
                'price_ratio_mean': ratio_mean,
                'price_ratio_std': ratio_std,
                'spread_volatility': spread_volatility,
                'data_points': len(data),
                'is_cointegrated': eg_result.is_cointegrated,
                'hedge_ratio': eg_result.hedge_ratio,
                'half_life': eg_result.half_life
            }
            
        except Exception as e:
            print(f"Cointegration test failed: {e}")
            return {
                'engle_granger': None,
                'johansen': None,
                'correlation': 0.0,
                'price_ratio_mean': 1.0,
                'price_ratio_std': 0.0,
                'spread_volatility': 0.0,
                'data_points': 0,
                'is_cointegrated': False,
                'hedge_ratio': 1.0,
                'half_life': None
            }


class PairValidator:
    """Validate pairs for trading suitability."""
    
    def __init__(self, min_correlation: float = 0.5, max_correlation: float = 0.95,
                 min_data_points: int = 100, max_spread_volatility: float = 0.5):
        """
        Initialize pair validator.
        
        Args:
            min_correlation: Minimum correlation threshold
            max_correlation: Maximum correlation threshold (avoid perfect correlation)
            min_data_points: Minimum data points required
            max_spread_volatility: Maximum allowed spread volatility
        """
        self.min_correlation = min_correlation
        self.max_correlation = max_correlation
        self.min_data_points = min_data_points
        self.max_spread_volatility = max_spread_volatility
        self.cointegration_tester = CointegrationTester()
    
    def validate_pair(self, price1: pd.Series, price2: pd.Series, 
                     ticker1: str, ticker2: str) -> Dict:
        """
        Validate a trading pair against all criteria.
        
        Args:
            price1, price2: Price series
            ticker1, ticker2: Ticker symbols
            
        Returns:
            Dictionary with validation results
        """
        # Test cointegration
        coint_results = self.cointegration_tester.test_pair_cointegration(price1, price2)
        
        # Validation checks
        checks = {
            'sufficient_data': coint_results['data_points'] >= self.min_data_points,
            'correlation_in_range': (
                self.min_correlation <= abs(coint_results['correlation']) <= self.max_correlation
            ),
            'cointegrated': coint_results['is_cointegrated'],
            'spread_volatility_ok': coint_results['spread_volatility'] <= self.max_spread_volatility,
            'has_half_life': coint_results['half_life'] is not None
        }
        
        # Overall validation
        is_valid = all(checks.values())
        
        # Calculate trading score
        score = self._calculate_trading_score(coint_results, checks)
        
        return {
            'pair': f"{ticker1}-{ticker2}",
            'is_valid': is_valid,
            'score': score,
            'checks': checks,
            'cointegration_results': coint_results,
            'validation_summary': self._create_summary(checks, coint_results)
        }
    
    def _calculate_trading_score(self, coint_results: Dict, checks: Dict) -> float:
        """Calculate a trading attractiveness score (0-100)."""
        score = 0.0
        
        # Cointegration (40 points)
        if checks['cointegrated']:
            score += 40
            # Bonus for strong statistical significance
            if coint_results['engle_granger'] and coint_results['engle_granger'].p_value < 0.01:
                score += 10
        
        # Correlation (20 points)
        if checks['correlation_in_range']:
            corr = abs(coint_results['correlation'])
            # Optimal correlation around 0.7-0.8
            if 0.7 <= corr <= 0.8:
                score += 20
            else:
                score += 15
        
        # Half-life (15 points)
        if checks['has_half_life']:
            half_life = coint_results['half_life']
            # Optimal half-life: 1-30 days
            if 1 <= half_life <= 30:
                score += 15
            elif half_life <= 60:
                score += 10
            else:
                score += 5
        
        # Data quality (15 points)
        if checks['sufficient_data']:
            data_points = coint_results['data_points']
            if data_points >= 250:
                score += 15
            elif data_points >= 150:
                score += 10
            else:
                score += 5
        
        # Spread volatility (10 points)
        if checks['spread_volatility_ok']:
            vol = coint_results['spread_volatility']
            if vol <= 0.2:
                score += 10
            elif vol <= 0.3:
                score += 7
            else:
                score += 3
        
        return min(score, 100.0)
    
    def _create_summary(self, checks: Dict, coint_results: Dict) -> str:
        """Create human-readable validation summary."""
        summary_parts = []
        
        if checks['cointegrated']:
            summary_parts.append("✓ Cointegrated")
        else:
            summary_parts.append("✗ Not cointegrated")
        
        corr = coint_results['correlation']
        summary_parts.append(f"Correlation: {corr:.3f}")
        
        if coint_results['half_life']:
            summary_parts.append(f"Half-life: {coint_results['half_life']:.1f} days")
        
        summary_parts.append(f"Data points: {coint_results['data_points']}")
        
        return " | ".join(summary_parts)
