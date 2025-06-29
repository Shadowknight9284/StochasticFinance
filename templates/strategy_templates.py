"""
Strategy Templates for Common Quantitative Models
================================================

This module contains templates for frequently used quantitative trading strategies
with their mathematical frameworks and implementation patterns.
"""

STRATEGY_TEMPLATES = {
    "ornstein_uhlenbeck": {
        "name": "Ornstein-Uhlenbeck Mean Reversion",
        "sde": "dX_t = \\theta(\\mu - X_t)dt + \\sigma dW_t",
        "description": "Classic mean-reversion model with exponential decay to long-term mean",
        "asset_classes": ["Equities", "FX", "Interest Rates"],
        "key_parameters": ["theta", "mu", "sigma"],
        "theoretical_properties": [
            "Stationary distribution: N(μ, σ²/(2θ))",
            "Half-life: ln(2)/θ", 
            "Autocorrelation: exp(-θτ)"
        ]
    },
    
    "heston": {
        "name": "Heston Stochastic Volatility",
        "sde": [
            "dS_t = rS_t dt + \\sqrt{V_t}S_t dW_t^S",
            "dV_t = \\kappa(\\theta - V_t)dt + \\sigma\\sqrt{V_t}dW_t^V"
        ],
        "description": "Two-factor model with stochastic volatility and correlation",
        "asset_classes": ["Options", "Volatility Surface"],
        "key_parameters": ["kappa", "theta", "sigma", "rho"],
        "theoretical_properties": [
            "Volatility clustering",
            "Leverage effect via correlation",
            "Semi-closed form option pricing"
        ]
    },
    
    "jump_diffusion": {
        "name": "Merton Jump Diffusion",
        "sde": "dS_t = \\mu S_t dt + \\sigma S_t dW_t + S_t dJ_t",
        "description": "Brownian motion with compound Poisson jumps",
        "asset_classes": ["Equity Indices", "Individual Stocks"],
        "key_parameters": ["mu", "sigma", "lambda", "mu_J", "sigma_J"],
        "theoretical_properties": [
            "Fat-tailed return distribution",
            "Volatility smile in options",
            "Closed-form European option pricing"
        ]
    },
    
    "vasicek": {
        "name": "Vasicek Interest Rate",
        "sde": "dr_t = a(b - r_t)dt + \\sigma dW_t",
        "description": "Single-factor short rate model with mean reversion",
        "asset_classes": ["Bonds", "Interest Rate Derivatives"],
        "key_parameters": ["a", "b", "sigma"],
        "theoretical_properties": [
            "Gaussian interest rates",
            "Affine term structure",
            "Closed-form bond pricing"
        ]
    },
    
    "cir": {
        "name": "Cox-Ingersoll-Ross",
        "sde": "dr_t = a(b - r_t)dt + \\sigma\\sqrt{r_t}dW_t",
        "description": "Square-root diffusion ensuring positive rates",
        "asset_classes": ["Bonds", "Interest Rate Derivatives"],
        "key_parameters": ["a", "b", "sigma"],
        "theoretical_properties": [
            "Always positive rates (if 2ab ≥ σ²)",
            "Chi-squared distribution",
            "Affine term structure"
        ]
    },
    
    "regime_switching": {
        "name": "Markov Regime Switching",
        "sde": "dS_t = \\mu(X_t)S_t dt + \\sigma(X_t)S_t dW_t",
        "description": "Parameters switch according to hidden Markov chain",
        "asset_classes": ["Equities", "Commodities", "FX"],
        "key_parameters": ["mu_1", "mu_2", "sigma_1", "sigma_2", "P"],
        "theoretical_properties": [
            "Multiple volatility regimes",
            "Regime persistence",
            "Non-linear filtering for state estimation"
        ]
    },
    
    "fractional_brownian": {
        "name": "Fractional Brownian Motion",
        "sde": "dS_t = \\mu S_t dt + \\sigma S_t dB_t^H",
        "description": "Long-memory process with Hurst parameter H",
        "asset_classes": ["Long-term trends", "Commodity prices"],
        "key_parameters": ["mu", "sigma", "H"],
        "theoretical_properties": [
            "Long-range dependence (H > 0.5)",
            "Self-similarity",
            "Non-Markovian"
        ]
    },
    
    "levy_alpha_stable": {
        "name": "Alpha-Stable Lévy Process",
        "sde": "dS_t = \\mu S_t dt + S_t dL_t^{\\alpha,\\beta}",
        "description": "Heavy-tailed jumps with infinite variance",
        "asset_classes": ["High-frequency data", "Crisis periods"],
        "key_parameters": ["alpha", "beta", "gamma", "delta"],
        "theoretical_properties": [
            "Power-law tails",
            "Stable distributions",
            "Infinite variance (α < 2)"
        ]
    }
}

IMPLEMENTATION_PATTERNS = {
    "mean_reversion": {
        "signal_logic": """
        // Z-score based signal
        double z_score = (current_price - long_term_mean) / volatility;
        if (z_score < -threshold) return BUY;
        if (z_score > threshold) return SELL;
        return HOLD;
        """,
        "risk_management": "Stop-loss at 2σ deviation",
        "position_sizing": "Kelly criterion or constant fraction"
    },
    
    "momentum": {
        "signal_logic": """
        // Trend following
        double momentum = (current_price - price_lagged) / price_lagged;
        if (momentum > entry_threshold) return BUY;
        if (momentum < -entry_threshold) return SELL;
        return HOLD;
        """,
        "risk_management": "Trailing stop or time-based exit",
        "position_sizing": "Volatility-adjusted position size"
    },
    
    "volatility_targeting": {
        "signal_logic": """
        // Volatility forecast based sizing
        double vol_forecast = estimate_volatility();
        double target_vol = 0.15; // 15% annual
        double position_scale = target_vol / vol_forecast;
        return base_signal * position_scale;
        """,
        "risk_management": "Dynamic position sizing",
        "position_sizing": "Inverse volatility weighting"
    }
}

def get_template(strategy_type: str) -> dict:
    """Get template for a specific strategy type."""
    return STRATEGY_TEMPLATES.get(strategy_type, {})

def list_templates() -> list:
    """List all available strategy templates."""
    return list(STRATEGY_TEMPLATES.keys())

def get_implementation_pattern(pattern_type: str) -> dict:
    """Get implementation pattern for signal generation."""
    return IMPLEMENTATION_PATTERNS.get(pattern_type, {})
