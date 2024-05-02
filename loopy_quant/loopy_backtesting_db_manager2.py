# import os
import streamlit as st

import pandas as pd
from sqlalchemy import text #, create_engine
# from sqlalchemy.orm import sessionmaker

from quants_lab.strategy.strategy_analysis import StrategyAnalysis
from loopy_quant.loopy_database_manager import LoopyDBManager


class LoopyBacktestingDBManager2(LoopyDBManager):
    def get_strategy_analysis(self, strategy=None, exchange=None, trading_pair=None, start_date=None, end_date=None):
        if "postgres" in self.db_name:
            if "backtesting" in self.db_name:
                try:
                    candle_data = self.get_candle_data(exchange=exchange, trading_pair=trading_pair, start_date=start_date, end_date=end_date)
                except Exception as e:
                    print(e)
                    candle_data = None
                try:
                    position_executor = self.get_backtesting_executor_data(strategy=strategy, exchange=exchange, trading_pair=trading_pair, start_date=start_date, end_date=end_date)
                except Exception as e:
                    print(e)
                    position_executor = None

        if candle_data is not None and position_executor is not None:
            # return LoopyStrategyAnalysis(positions=position_executor, candles_df=candle_data)
            return StrategyAnalysis(positions=position_executor, candles_df=candle_data)
        else:
            return None
    
    @staticmethod
    def _get_backtesting_executors_query(strategy=None, exchange=None, trading_pair=None, start_date=None, end_date=None):
        query = "SELECT * FROM backtesting_executors"
        conditions = []
        if strategy:
            conditions.append(f"strategy = '{strategy}'")
        if exchange:
            conditions.append(f"exchange = '{exchange}'")
        if trading_pair:
            conditions.append(f"symbol = '{trading_pair}'")
        if start_date:
            conditions.append(f"timestamp >= '{start_date}'")
        if end_date:
            conditions.append(f"timestamp <= '{end_date}'")
        if conditions:
            query += f" WHERE {' AND '.join(conditions)}"
        return query
    
    @staticmethod
    def _get_backtesting_strategy_query():
        query = "SELECT strategy FROM backtesting_result group by strategy"
        return query
    
    @staticmethod
    def _get_backtesting_strategy_results_query(strategy):
        query = f"SELECT * FROM backtesting_result WHERE strategy = '{strategy}'"
        return query
    
    def get_backtesting_executor_data(self, strategy=None, exchange=None, trading_pair=None, start_date=None, end_date=None) -> pd.DataFrame:
        with self.session_maker() as session:
            query = self._get_backtesting_executors_query(strategy=strategy,exchange=exchange,trading_pair=trading_pair,start_date=start_date, end_date=end_date)
            try:
                backtesting_executor = pd.read_sql_query(text(query), session.connection())
            except Exception as e:
                print(e)
                return None
            
            columns = [
                "timestamp",
                "exchange",
                "trading_pair",
                "amount",
                "side",
                "trade_pnl",
                "trade_pnl_quote",
                "cum_fee_quote",
                "net_pnl_quote",
                "net_pnl",
                "profitable",
                "inventory",
                "close_timestamp",
                "executor_status",
                "close_type",
                "close",
                "entry_price",
                "close_price",
                "sl",
                "tp",
                "tl",
                "open_order_type",
                "take_profit_order_type",
                "stop_loss_order_type",
                "time_limit_order_type",
                "leverage",
                "controller_name"
            ]
            position_executor = pd.DataFrame(columns=columns)

            position_executor["timestamp"] = backtesting_executor["timestamp"]
            position_executor["exchange"] = backtesting_executor["exchange"]
            position_executor["trading_pair"] = backtesting_executor["symbol"]
            position_executor["amount"] = backtesting_executor["amount"]
            position_executor["side"] = backtesting_executor["side"]
            position_executor["trade_pnl"] = backtesting_executor["trade_pnl"]
            position_executor["trade_pnl_quote"] = 0
            position_executor["cum_fee_quote"] = 0
            position_executor["net_pnl_quote"] = backtesting_executor["net_pnl_quote"]
            position_executor["net_pnl"] = backtesting_executor["net_pnl"]
            position_executor["profitable"] = backtesting_executor["profitable"]
            position_executor["inventory"] = backtesting_executor["inventory"]
            position_executor["close_time"] = backtesting_executor["close_time"]
            position_executor["executor_status"] = None
            position_executor["close_type"] = backtesting_executor["close_type"]
            position_executor["close"] = backtesting_executor["close"]
            position_executor["entry_price"] = backtesting_executor["close"]
            position_executor["close_price"] = backtesting_executor["close_price"]
            position_executor["sl"] = backtesting_executor["sl"]
            position_executor["tp"] = backtesting_executor["tp"]
            # position_executor["tl"] = backtesting_executor["tl"]
            position_executor["tl"] = backtesting_executor["tl"].astype('int64') / 1e9
            position_executor["open_order_type"] = None
            position_executor["take_profit_order_type"] = None
            position_executor["stop_loss_order_type"] = None
            position_executor["time_limit_order_type"] = None
            position_executor["leverage"] = 1
            position_executor["controller_name"] = backtesting_executor["strategy"]
            
            position_executor["level"] = backtesting_executor["order_level"].apply(lambda x: x.split("_")[1])

            # position_executor.set_index("timestamp", inplace=True)
            position_executor.set_index("timestamp")
            position_executor["datetime"] = pd.to_datetime(position_executor.index, unit="s")
            position_executor["close_datetime"] = pd.to_datetime(position_executor["close_time"], unit="s")

            # Assigning an auto-increment 'id' column starting from 1
            position_executor["id"] = range(1, len(position_executor) + 1)

            # print("-last position_executor-----------")
            # print(position_executor.iloc[-1])

        return position_executor
    
    def get_backtesting_strategy_list(self):
        with self.session_maker() as session:
            query = self._get_backtesting_strategy_query()
            try:
                strategy_list = pd.read_sql_query(text(query), session.connection())
            except Exception as e:
                print(e)
                return None

        return strategy_list

    def get_backtesting_strategy_summary(self, strategy):
        with self.session_maker() as session:
            query = self._get_backtesting_strategy_results_query(strategy)
            try:
                result = pd.read_sql_query(text(query), session.connection())
            except Exception as e:
                print(e)
                return None
            
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
                "mdd_usd",
                "mdd_pct",
                "sharp_ratio",
                "profit_factor",
                "duration_hours",
                "avg_trading_hours",
                "win_signals",
                "loss_signals"
            ]
            strategy_summary = pd.DataFrame(columns=columns)
            # market_data["timestamp"] = pd.to_datetime(market_data["timestamp"] / 1e6, unit="ms")
            strategy_summary["timestamp"] = result["timestamp"]
            strategy_summary["strategy"] = result["strategy"]
            strategy_summary["strategy_id"] = result["strategy_id"]
            strategy_summary["exchange"] = result["exchange"]
            strategy_summary["trading_pair"] = result["symbol"]
            strategy_summary["interval"] = result["interval"]
            strategy_summary["start"] = result["start"]
            strategy_summary["end"] = result["end"]
            strategy_summary["initial_portfolio_usd"] = result["initial_portfolio_usd"]
            strategy_summary["trade_cost"] = result["trade_cost"]
            strategy_summary["sl"] = result["stop_loss"]
            strategy_summary["tp"] = result["take_profit"]
            strategy_summary["tl"] = result["time_limit"]
            strategy_summary["net_pnl"] = result["net_pnl"]
            strategy_summary["net_pnl_usd"] = result["net_pnl_quote"]
            strategy_summary["total_executors"] = result["total_executors"]
            strategy_summary["total_volume"] = result["total_volume"]
            strategy_summary["total_long"] = result["total_long"]
            strategy_summary["total_short"] = result["total_short"]
            strategy_summary["total_positions"] = result["total_positions"]
            strategy_summary["accuracy"] = result["accuracy"]
            strategy_summary["mdd_usd"] = result["max_drawdown_usd"]
            strategy_summary["mdd_pct"] = result["max_drawdown_pct"]
            strategy_summary["sharp_ratio"] = result["sharpe_ratio"]
            strategy_summary["profit_factor"] = result["profit_factor"]
            strategy_summary["duration_hours"] = result["duration_hours"]
            strategy_summary["avg_trading_hours"] = result["avg_trading_time_hours"]
            strategy_summary["win_signals"] = result["win_signals"]
            strategy_summary["loss_signals"] = result["loss_signals"]
            
            strategy_summary.set_index("timestamp", inplace=True)

            # print("-last market_data-----------")
            # print(market_data.iloc[-1])
            # print(market_data.iloc[-1]["timestamp"])

        return strategy_summary 
    
    def backtesting_strategy_list(self):
        strategy_list = self.get_backtesting_strategy_list()
        if strategy_list is not None:
            selected_strategy = st.selectbox("Choose a backtesting strategy:", strategy_list)
        else:
            selected_strategy = None

        return selected_strategy
        
    def backtesting_strategy_summary_table(self, strategy):
        strategy_summary = self.get_backtesting_strategy_summary(strategy)
        if strategy_summary is None:
            return None
        
        summary_table = st.data_editor(strategy_summary,
                                 column_config={"Explore": st.column_config.CheckboxColumn(required=True)},
                                 use_container_width=True,
                                 hide_index=True
                                 )
        selected_rows = summary_table[summary_table.Explore]
        if len(selected_rows) > 0:
            return selected_rows
        else:
            return None
        
    

    