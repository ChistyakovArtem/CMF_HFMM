import numpy as np


def SharpeRatio(pnl):
    daily_pnl = pnl.diff().dropna()
    return daily_pnl.mean() / daily_pnl.std()


def MaximumDailyDrawdown(pnl):
    Roll_Max = pnl.cummax()
    Daily_Drawdown = pnl/Roll_Max - 1.0
    Max_Daily_Drawdown = Daily_Drawdown.cummin()
    return Max_Daily_Drawdown


def MaximumDrawdown(pnl):
    # = min(1 - pnl(i)/pnl(j)) : j > i
    Roll_Max = pnl.cummax()
    Drawdown = pnl/Roll_Max - 1
    return Drawdown.min()


def GetMetrics(pnl):
    return {
        'SharpeRatio': SharpeRatio(pnl),
        'MaximumDailyDrawdown': MaximumDailyDrawdown(pnl),
        'MaximumDrawdown': MaximumDrawdown(pnl)
    }
