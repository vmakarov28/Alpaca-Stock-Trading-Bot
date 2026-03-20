## The 31 Technical Features Used by the Model

The neural network looks at **31 different features** every 15 minutes to decide whether the stock is likely to go up or down in the next ~5 hours. Here's what each one means and what it tells the model:

### Basic Price & Volume (Raw Market Data)
- **close** — Current closing price of the stock  
- **high** — Highest price reached in the 15-minute bar  
- **low** — Lowest price reached in the 15-minute bar  
- **volume** — How many shares were traded in that bar (shows activity level)

### Moving Averages & Trend
- **MA20** — 20-period Simple Moving Average (short-term trend)  
- **MA50** — 50-period Simple Moving Average (medium-term trend)  
- **Trend** — Simple flag: 1 if price is above MA20 (uptrend), 0 otherwise

### Momentum & Oscillators
- **RSI** — Relative Strength Index (14-period) — shows if the stock is overbought (>70) or oversold (<30)  
- **MACD** — Moving Average Convergence Divergence — measures momentum and trend changes  
- **MACD_signal** — 9-period signal line of MACD (used for crossovers)  
- **Stoch_K** — Fast Stochastic %K — shows where the current price is in the recent high/low range  
- **Stoch_D** — Slow Stochastic %D — smoother version of Stoch_K for confirmation  
- **ADX** — Average Directional Index — tells how strong the current trend is (higher = stronger trend)

### Volatility & Risk Measures
- **ATR** — Average True Range (14-period) — measures how much the stock typically moves per bar  
- **Volatility** — 20-period standard deviation of returns — shows recent price swings  
- **Close_ATR** — Price divided by ATR — shows how “expensive” the stock is relative to its normal movement  
- **MA20_ATR** — MA20 divided by ATR — helps spot when the trend is stretched

### Volume-Based Indicators
- **OBV** — On-Balance Volume — tracks whether volume is pushing price up or down over time  
- **VWAP** — Volume Weighted Average Price — average price weighted by volume (institutional benchmark)  
- **CMF** — Chaikin Money Flow — shows if money is flowing into or out of the stock  
- **Volume_Delta** — Volume × (close - open) — shows buying vs selling pressure in each bar  
- **VWAP_Dev** — How far the current price is from VWAP — useful for mean-reversion signals

### Advanced / Custom Features
- **BB_upper** — Upper Bollinger Band (20-period, 2 std devs) — upper volatility boundary  
- **BB_lower** — Lower Bollinger Band — lower volatility boundary  
- **Return_1d** — 1-bar percentage return  
- **Return_5d** — 5-bar percentage return (helps spot short-term momentum)  
- **RSI_60** — 60-minute timeframe RSI (multi-timeframe view)  
- **MA20_60** — 60-minute timeframe MA20 (longer-term context)  
- **Macro_Stress** — Ratio of recent volatility vs longer-term volatility (detects regime changes)  
- **Earnings_Proxy** — Sentiment × volume strength — combines news mood with trading activity  
- **Sentiment** — News sentiment score from DistilBERT (currently neutral by default)

These 31 features give the model a rich, multi-angle view of the market — price action, momentum, volatility, volume pressure, and even a hint of sentiment. This is why the bot can make smarter decisions than simple rule-based systems.
