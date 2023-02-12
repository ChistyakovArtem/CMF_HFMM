# CMF_HFMM
My work on a project "High Frequency Market Making: Optimal Quoting" in the Center of Mathematical Finances.

# Results
- Made a High Frequency Market Making strategy that provides **0.9 million $ of liquidity in a day and earns 35% daily yield with 2 % MDD**.
- This strategy can be found in https://github.com/ChistyakovArtem/CMF_HFMM/blob/main/TechCore/Strategies/ML_Strategies/ML_Stoikov.py.
- Fitting of this straategy - here https://github.com/ChistyakovArtem/CMF_HFMM/blob/main/Notebooks/ML_1-Stoikov-with-ML(Model-Fitter).ipynb.
- Running process and results - here https://github.com/ChistyakovArtem/CMF_HFMM/blob/main/Notebooks/ML_1-Stoikov-with-ML.ipynb.

# Strategy description
- Baseline of this strategy is a legendary https://math.nyu.edu/~avellane/HighFrequencyTrading.pdf article that allows to handle two market making risks - Adverse selection (if price is constantly decreasing - only bid orders will be executed - so we will only buy asset - resulting a loss) and Inventory risk - if we have large inventory - we will have a risk of losses due to price changes.

- However this strategy operates with current market statistics: midprice, order_intensity and volatility. But I will try to look in the future.
- This strategy involves a Light GBM with **linear_tree** and **early_stopping** to predict future **return** (**500ms horizon - correlation 0.19**) (by predicting return), future **order_intensity** (**2s horizon - correlation - 0.07**) and **future volatility** (**1s horizon - correlation 0.16**) based on previous orderbook states.

- Total pnl line is very stable.
