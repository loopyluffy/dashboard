from typing import Optional
import datetime

import pandas as pd
from plotly.subplots import make_subplots
import pandas_ta as ta  # noqa: F401
import plotly.graph_objs as go
import numpy as np

from quants_lab.strategy.strategy_analysis import StrategyAnalysis


class LoopyPositionAnalysis(StrategyAnalysis):
    def get_single_Position_analysis(self, exchange: str, trading_pair: str):
        if self.candles_df is not None:
            candles_df = self.candles_df[(self.candles_df["exchange"] == exchange) &
                                         (self.candles_df["trading_pair"] == trading_pair)].copy()
        else:
            candles_df = None
        positions = self.positions[(self.positions["exchange"] == exchange) &
                                   (self.positions["trading_pair"] == trading_pair)].copy()

        return LoopyPositionAnalysis(
            positions=positions,
            candles_df=candles_df
        )
    
    def get_filtered_position_data(self, start_date: datetime.datetime, end_date: datetime.datetime):
        if self.candles_df is not None:
            candles_df = self.candles_df[(self.candles_df["timestamp"] >= start_date) &
                                         (self.candles_df["timestamp"] <= end_date)].copy()
        else:
            candles_df = None
        positions = self.positions[(self.positions["timestamp"] >= start_date) &
                                   (self.positions["timestamp"] <= end_date)].copy()

        return LoopyPositionAnalysis(
            positions=positions,
            candles_df=candles_df
        )
        
    def get_filtered_single_position_data(self, exchange: str, trading_pair: str, start_date: datetime.datetime, end_date: datetime.datetime):
        if self.candles_df is not None:
            candles_df = self.candles_df[(self.candles_df["exchange"] == exchange) &
                                         (self.candles_df["trading_pair"] == trading_pair) &
                                         (self.candles_df["timestamp"] >= start_date) &
                                         (self.candles_df["timestamp"] <= end_date)].copy()
        else:
            candles_df = None
        positions = self.positions[(self.positions["exchange"] == exchange) &
                                   (self.positions["trading_pair"] == trading_pair) &
                                   (self.positions["timestamp"] >= start_date) &
                                   (self.positions["timestamp"] <= end_date)].copy()

        return LoopyPositionAnalysis(
            positions=positions,
            candles_df=candles_df
        )
    
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
        
    def get_strategy_summary(self):
        # trade fill summary @backitesting db --------------------------------
        trade_fills["cum_fees_in_quote"] = trade_fills.groupby(groupers)["trade_fee_in_quote"].cumsum()
        # trade_fills["net_amount"] = trade_fills['amount'] * trade_fills['trade_type'].apply(lambda x: 1 if x == 'BUY' else -1)
        # trade_fills["net_amount_quote"] = trade_fills['net_amount'] * trade_fills['price']
        trade_fills["cum_net_amount"] = trade_fills.groupby(groupers)["net_amount"].cumsum()
        trade_fills["unrealized_trade_pnl"] = -1 * trade_fills.groupby(groupers)["net_amount_quote"].cumsum()
        trade_fills["inventory_cost"] = trade_fills["cum_net_amount"] * trade_fills["price"]
        trade_fills["realized_trade_pnl"] = trade_fills["unrealized_trade_pnl"] + trade_fills["inventory_cost"]
        trade_fills["net_realized_pnl"] = trade_fills["realized_trade_pnl"] - trade_fills["cum_fees_in_quote"]
        trade_fills["realized_pnl"] = trade_fills.groupby(groupers)["net_realized_pnl"].diff()
        trade_fills["gross_pnl"] = trade_fills.groupby(groupers)["realized_trade_pnl"].diff()
        trade_fills["trade_fee"] = trade_fills.groupby(groupers)["cum_fees_in_quote"].diff()
        # -------------------------------------------------------------------

        # trade fill summary @real db ---------------------------------------
        groupers = ["market", "symbol"]
        trade_fills["cum_fees_in_quote"] = trade_fills.groupby(groupers)["trade_fee_in_quote"].cumsum()
        trade_fills["cum_net_amount"] = trade_fills.groupby(groupers)["net_amount"].cumsum()
        if contract_multiplier == 1:
            # spot or vanilla contract
            trade_fills["unrealized_trade_pnl"] = -1 * trade_fills.groupby(groupers)["net_amount_quote"].cumsum()
            trade_fills["inventory_cost"] = trade_fills["cum_net_amount"] * trade_fills["price"]
        else:
            # inverse contract
            # trade_fills["unrealized_trade_pnl"] = trade_fills.groupby(groupers)["net_amount_quote"].cumsum()
            # trade_fills["inventory_cost"] = -1 * trade_fills["cum_net_amount"] * trade_fills["price"]
            trade_fills["unrealized_trade_pnl"] = trade_fills["cum_net_amount"] * trade_fills["price"]
            trade_fills["inventory_cost"] = -1 * trade_fills.groupby(groupers)["net_amount_quote"].cumsum()
        trade_fills["realized_trade_pnl"] = trade_fills["unrealized_trade_pnl"] + trade_fills["inventory_cost"]
        trade_fills["net_realized_pnl"] = trade_fills["realized_trade_pnl"] - trade_fills["cum_fees_in_quote"]
        trade_fills["realized_pnl"] = trade_fills.groupby(groupers)["net_realized_pnl"].diff()
        trade_fills["gross_pnl"] = trade_fills.groupby(groupers)["realized_trade_pnl"].diff()
        trade_fills["trade_fee"] = trade_fills.groupby(groupers)["cum_fees_in_quote"].diff()
        # -------------------------------------------------------------------

        # backtesting summary -----------------------------------------------
        row["trade_pnl"] = (close_price / row["close"] - 1)  * row["signal"]
        # (px.loc[df["close_time"].values].values / px.loc[df.index] - 1) * df["signal"]
        row["net_pnl"] = row["trade_pnl"] - trade_cost
        # row["net_pnl_quote"] = row["net_pnl"] * row["amount"]
        row["net_pnl_quote"] = row["net_pnl"] * float(order_level.order_amount_usd) * margin_multiplier

        net_pnl = executors_df["net_pnl"].sum()
        net_pnl_quote = executors_df["net_pnl_quote"].sum()
        total_volume = executors_with_position["amount"].sum() * 2
        accuracy = win_signals.shape[0] / total_positions
        cumulative_returns = executors_df["net_pnl_quote"].cumsum()
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak)
        max_draw_down = np.min(drawdown)
        max_drawdown_pct = max_draw_down / executors_df["inventory"].iloc[0]
        returns = executors_df["net_pnl_quote"] / net_pnl
        sharpe_ratio = returns.mean() / returns.std()
        total_won = win_signals.loc[:, "net_pnl_quote"].sum()
        total_loss = - loss_signals.loc[:, "net_pnl_quote"].sum()
        profit_factor = total_won / total_loss if total_loss > 0 else 1

        return {
            "net_pnl": net_pnl,
            "net_pnl_quote": net_pnl_quote,
            "total_executors": total_executors,
            "total_executors_with_position": total_executors_with_position,
            "total_volume": total_volume,
            "total_long": total_long,
            "total_short": total_short,
            "close_types": close_types,
            "accuracy_long": accuracy_long,
            "accuracy_short": accuracy_short,
            "total_positions": total_positions,
            "accuracy": accuracy,
            "max_drawdown_usd": max_draw_down,
            "max_drawdown_pct": max_drawdown_pct,
            "sharpe_ratio": sharpe_ratio,
            "profit_factor": profit_factor,
            "duration_minutes": duration_minutes,
            "avg_trading_time_minutes": avg_trading_time,
            "win_signals": win_signals.shape[0],
            "loss_signals": loss_signals.shape[0],
        }
        # -------------------------------------------------------------------

        # backtesting summary -----------------------------------------------
        columns = [
                "timestamp",
                "strategy",
                "strategy_id",
                "exchange",
                "trading_pair",
                "interval",
                "start",
                "end",
                "initial_portfolio_usd",
                "trade_cost",
                "sl",
                "tp",
                "tl",
                "net_pnl",
                "net_pnl_usd",
                "total_executors",
                "total_volume",
                "total_long",
                "total_short",
                "total_positions",
                "accuracy",
                "max_drawdown_usd",
                "max_drawdown_pct",
                "sharp_ratio",
                "profit_factor",
                "duration_hours",
                "avg_trading_hours",
                "win_signals",
                "loss_signals"
            ]
        # ----------------------------------------------------


        columns_dict = {"strategy": "Strategy",
                        "market": "Exchange",
                        "symbol": "Trading Pair",
                        "order_id_count": "# Trades",
                        "total_positions": "# Positions",
                        "volume_sum": "Volume",
                        "TAKE_PROFIT": "# TP",
                        "STOP_LOSS": "# SL",
                        "TRAILING_STOP": "# TSL",
                        "TIME_LIMIT": "# TL",
                        "net_realized_pnl_full_series": "PnL Over Time",
                        "net_realized_pnl_last": "Realized PnL"}

        pos_data = self.positions.copy()
        grouped_positions = pos_data.groupby(
            ["exchange", "trading_pair", "controller_name", "close_type"]).agg(
            metric_count=("close_type", "count")).reset_index()
        index_cols = ["exchange", "trading_pair", "controller_name"]
        pivot_positions = pd.pivot_table(grouped_positions, values="metric_count", index=index_cols,
                                            columns="close_type").reset_index()
        result_cols = ["TAKE_PROFIT", "STOP_LOSS", "TRAILING_STOP", "TIME_LIMIT"]
        pivot_positions = pivot_positions.reindex(columns=index_cols + result_cols, fill_value=0)
        pivot_positions["total_positions"] = pivot_positions[result_cols].sum(axis=1)
        strategy_summary = grouped_trade_fill.merge(pivot_executors, left_on=["market", "symbol"],
                                                    right_on=["exchange", "trading_pair"],
                                                    how="left")
        strategy_summary.drop(columns=["exchange", "trading_pair"], inplace=True)


class LoopySinglePositionAnalysis(StrategyAnalysis):    
    def __init__(self, exchange: str, trading_pair: str, positions: pd.DataFrame, candles_df: Optional[pd.DataFrame] = None):
        super().__init__(positions=positions, candles_df=candles_df)
        self.exchange = exchange
        self.trading_pair = trading_pair

#     def __init__(self, positions: pd.DataFrame, candles_df: Optional[pd.DataFrame] = None):
#         self.candles_df = candles_df
#         self.positions = positions
#         self.candles_df["timestamp"] = pd.to_datetime(self.candles_df["timestamp"], unit="ms")
#         self.positions["timestamp"] = pd.to_datetime(self.positions["timestamp"], unit="ms")
#         self.positions["close_time"] = pd.to_datetime(self.positions["close_time"], unit="ms")
#         self.base_figure = None

#     def initial_portfolio(self):
#         return self.positions["inventory"].dropna().values[0]

#     def final_portfolio(self):
#         return self.positions["inventory"].dropna().values[-1]

#     def net_profit_usd(self):
#         # TODO: Fix inventory calculation, gives different to this
#         return self.positions["net_pnl_quote"].sum()

#     def net_profit_pct(self):
#         return self.net_profit_usd() / self.initial_portfolio()

#     def returns(self):
#         return self.positions["net_pnl_quote"] / self.initial_portfolio()

#     def total_positions(self):
#         # TODO: Determine if it has to be shape[0] - 1 or just shape[0]
#         return self.positions.shape[0] - 1

#     def win_signals(self):
#         return self.positions.loc[(self.positions["profitable"] > 0) & (self.positions["side"] != 0)]

#     def loss_signals(self):
#         return self.positions.loc[(self.positions["profitable"] < 0) & (self.positions["side"] != 0)]

#     def accuracy(self):
#         return self.win_signals().shape[0] / self.total_positions()

#     def max_drawdown_usd(self):
#         cumulative_returns = self.positions["net_pnl_quote"].cumsum()
#         peak = np.maximum.accumulate(cumulative_returns)
#         drawdown = (cumulative_returns - peak)
#         max_draw_down = np.min(drawdown)
#         return max_draw_down

#     def max_drawdown_pct(self):
#         return self.max_drawdown_usd() / self.initial_portfolio()

#     def sharpe_ratio(self):
#         returns = self.returns()
#         return returns.mean() / returns.std()

#     def profit_factor(self):
#         total_won = self.win_signals().loc[:, "net_pnl_quote"].sum()
#         total_loss = - self.loss_signals().loc[:, "net_pnl_quote"].sum()
#         return total_won / total_loss

#     def duration_in_minutes(self):
#         return (self.positions["timestamp"].iloc[-1] - self.positions["timestamp"].iloc[0]).total_seconds() / 60

#     def avg_trading_time_in_minutes(self):
#         time_diff_minutes = (self.positions["close_time"] - self.positions["timestamp"]).dt.total_seconds() / 60
#         return time_diff_minutes.mean()

#     def start_date(self):
#         return pd.to_datetime(self.candles_df.index.min(), unit="ms")

#     def end_date(self):
#         return pd.to_datetime(self.candles_df.index.max(), unit="ms")

#     def avg_profit(self):
#         return self.positions.net_pnl_quote.mean()

#     def text_report(self):
#         return f"""
# Strategy Performance Report:
#     - Net Profit: {self.net_profit_usd():,.2f} USD ({self.net_profit_pct() * 100:,.2f}%)
#     - Total Positions: {self.total_positions()}
#     - Win Signals: {self.win_signals().shape[0]}
#     - Loss Signals: {self.loss_signals().shape[0]}
#     - Accuracy: {self.accuracy():,.2f}%
#     - Profit Factor: {self.profit_factor():,.2f}
#     - Max Drawdown: {self.max_drawdown_usd():,.2f} USD | {self.max_drawdown_pct() * 100:,.2f}%
#     - Sharpe Ratio: {self.sharpe_ratio():,.2f}
#     - Duration: {self.duration_in_minutes() / 60:,.2f} Hours
#     - Average Trade Duration: {self.avg_trading_time_in_minutes():,.2f} minutes
#     """

#     def pnl_over_time(self):
#         fig = go.Figure()
#         positions = self.positions.copy()
#         positions.reset_index(inplace=True).sort_values("close_time", inplace=True)
#         fig.add_trace(go.Scatter(name="PnL Over Time",
#                                  x=self.positions.close_time,
#                                  y=self.positions.net_pnl_quote.cumsum()))
#         # Update layout with the required attributes
#         fig.update_layout(
#             title="PnL Over Time",
#             xaxis_title="N° Position",
#             yaxis=dict(title="Net PnL USD", side="left", showgrid=False),
#         )
#         return fig
