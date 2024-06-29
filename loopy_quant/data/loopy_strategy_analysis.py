from typing import Optional
import datetime

import pandas as pd
from plotly.subplots import make_subplots
import pandas_ta as ta  # noqa: F401
import plotly.graph_objs as go
import numpy as np

from quants_lab.strategy.strategy_analysis import StrategyAnalysis


class LoopyPositionAnalysis(StrategyAnalysis):
    # StrategyAnalysis Attributes -------------------
    # initial_portfolio
    # final_portfolio
    # net_profit_usd
    # net_profit_pct
    # returns
    # total_positions
    # win_signals
    # loss_signals
    # accuracy
    # max_drawdown_usd
    # max_drawdown_pct
    # sharpe_ratio
    # profit_factor
    # duration_in_minutes
    # avg_trading_time_in_minutes
    # start_date
    # end_date
    # avg_profit
    # pnl_over_time

    def total_positions(self):
        # TODO: Determine if it has to be shape[0] - 1 or just shape[0]
        return self.positions.shape[0]
    
    def net_profit(self):
        return self.positions["net_pnl"].sum()

    def total_volume(self):
        return self.positions["amount"].sum() * 2
    
    def total_long(self):
        return (lambda x: (x["side"] == 'BUY').sum() if x["side"].dtype == object else (x["side"] == 1).sum())(self.positions)
    
    def total_short(self):
        return (lambda x: (x["side"] == 'SELL').sum() if x["side"].dtype == object else (x["side"] == -1).sum())(self.positions)
    
    def correct_long(self):
        return (lambda x: (x["side"] == 'BUY' & x["net_pnl"] > 0).sum() if x["side"].dtype == object else (x["side"] == 1 & x["net_pnl"] > 0).sum())(self.positions)
    
    def correct_short(self):
        return (lambda x: (x["side"] == 'SELL' & x["net_pnl"] > 0).sum() if x["side"].dtype == object else (x["side"] == -1 & x["net_pnl"] > 0).sum())(self.positions)

    def accuracy_long(self):
        total_long = self.total_long()
        return self.correct_long() / total_long if total_long > 0 else 0
    
    def accuracy_short(self):
        total_short = self.total_short()
        return self.correct_short() / total_short if total_short > 0 else 0
    
    def close_types(self):
        return self.positions.groupby("close_type")["timestamp"].count()
    
    def get_strategy_summary(self):
        return {
            # "start": self.start_date(),
            # "end": self.end_date(),
            "initial_portfolio_usd": self.initial_portfolio(),
            # "trade_cost",
            "net_pnl": self.net_profit(),
            "net_pnl_quote": self.net_profit_usd(),
            # "total_executors": total_executors,
            # "total_executors_with_position": total_executors_with_position,
            "total_volume": self.total_volume(),
            "total_long": self.total_long(),
            "total_short": self.total_short(),
            "close_types": self.close_types(),
            "accuracy_long": self.accuracy_long(),
            "accuracy_short": self.accuracy_short(),
            "total_positions": self.total_positions(),
            "accuracy": self.accuracy(),
            "max_drawdown_usd": self.max_draw_down_usd(),
            "max_drawdown_pct": self.max_drawdown_pct(),
            "sharpe_ratio": self.sharpe_ratio(),
            "profit_factor": self.profit_factor(),
            "duration_minutes": self.duration_in_minutes(),
            "avg_trading_time_minutes": self.avg_trading_time_in_minutes(),
            "win_signals": self.win_signals().shape[0],
            "loss_signals": self.loss_signals().shape[0],
        }
    
    def get_filtered_position_analysis(self, exchange: str = None, trading_pair: str = None, 
                                   start_date: datetime = None, end_date: datetime = None) -> 'StrategyAnalysis':
        if self.candles_df:
            filtered_candles = self.candles_df.copy()
            if exchange:
                filtered_candles = filtered_candles[(filtered_candles["exchange"] == exchange)]
            if trading_pair:
                filtered_candles = filtered_candles[(filtered_candles["trading_pair"] == trading_pair)]
            if start_date:
                filtered_candles = filtered_candles[filtered_candles['timestamp'] >= start_date]
            if end_date:
                filtered_candles = filtered_candles[filtered_candles['timestamp'] <= end_date]
        else:
            filtered_candles = None

        if self.positions:
            filtered_positions = self.positions.copy()
            if exchange:
                filtered_positions = filtered_positions[filtered_positions['exchange'] == exchange]
            if trading_pair:
                filtered_positions = filtered_positions[filtered_positions['trading_pair'] == trading_pair]
            if start_date:
                filtered_positions = filtered_positions[filtered_positions['timestamp'] >= start_date]
            if end_date:
                filtered_positions = filtered_positions[filtered_positions['timestamp'] <= end_date]
        else:
            return None

        # Create a new StrategyAnalysis instance with filtered data
        return StrategyAnalysis(positions=filtered_positions, candles_df=filtered_candles)
    
    # interval is on a second unit
    def get_candles_resampled(self, interval: int):
        # Check if the interval is less than or equal to 5 minutes
        if interval <= (60 * 5):
            return self.candles_df
        else:
            # Assuming self.candles_df has a datetime index
            # if not self.candles_df.index.is_all_dates:
            #     raise ValueError("DataFrame index must be datetime type for resampling.")
            
            # Perform resampling and aggregate, here using the last as an example
            return self.candles_df.resample(f"{interval}S").last()
        

# class LoopySinglePositionAnalysis(StrategyAnalysis):    
#     def __init__(self, exchange: str, trading_pair: str, positions: pd.DataFrame, candles_df: Optional[pd.DataFrame] = None):
#         super().__init__(positions=positions, candles_df=candles_df)
#         self.exchange = exchange
#         self.trading_pair = trading_pair

