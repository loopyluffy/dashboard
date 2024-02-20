import os
import streamlit as st

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# from utils.data_manipulation import StrategyData
from loopy_quant.loopy_data_manipulation import LoopyStrategyData

from loopy_quant.loopy_database_manager import LoopyDBManager


class LoopyBacktestingDBManager(LoopyDBManager):
    def get_strategy_data(self, config_file_path=None, start_date=None, end_date=None):
        def load_data(table_loader, start_date=None, end_date=None):
            try:
                return table_loader(start_date, end_date)
            except Exception as e:
                print(e)
                return None  # Return None to indicate failure
            
        strategy_data = None

        # Use load_data to load tables
        if "postgres" in self.db_name:
            if "backtesting" in self.db_name:
                market_data = load_data(self.get_market_data, self.start_date, self.end_date)
                trade_fills = load_data(self.get_backtesting_trade_fills, self.start_date, self.end_date)
                # position_executor = None
                position_executor = load_data(self.get_backtesting_executor_data, self.start_date, self.end_date)
                strategy_data = LoopyStrategyData(None, None, trade_fills, market_data, position_executor)
           
        return strategy_data
    
    @staticmethod
    def _get_table_status(table_loader, start_date=None, end_date=None):
        try:
            data = table_loader(start_date, end_date)
            return "Correct" if len(data) > 0 else f"Error - No records matched"
        except Exception as e:
            return f"Error - {str(e)}"

    @property
    def status(self):
        status = None
        if "postgres" in self.db_name:
            if "backtesting" in self.db_name:
                status = {
                    "db_name": self.db_name,
                    "orders": "Correct",
                    "order_status": "Correct",
                    "trade_fill": self._get_table_status(self.get_backtesting_trade_fills, self.start_date, self.end_date),
                    "market_data": self._get_table_status(self.get_market_data, self.start_date, self.end_date),
                    "position_executor": self._get_table_status(self.get_backtesting_executor_data, self.start_date, self.end_date),
                }
      
        return status

    @staticmethod
    # def _get_position_executor_query(start_date=None, end_date=None):
    def _get_backtesting_executors_query(start_date=None, end_date=None):
        query = "SELECT * FROM backtesting_executors"
        conditions = []
        if start_date:
            conditions.append(f"timestamp >= '{start_date}'")
        if end_date:
            conditions.append(f"timestamp <= '{end_date}'")
        if conditions:
            query += f" WHERE {' AND '.join(conditions)}"
        return query
    
    def get_backtesting_executor_data(self, start_date=None, end_date=None) -> pd.DataFrame:
        with self.session_maker() as session:
            query = self._get_backtesting_executors_query(start_date, end_date)
            try:
                backtesting_executor = pd.read_sql_query(text(query), session.connection())
            except Exception as e:
                print(e)
                return None
            
            columns = ["timestamp",
                        "exchange",
                        "trading_pair"
                        "amount",
                        "side",
                        "trade_pnl",
                        "trade_pnl_quote",
                        "cum_fee_quote",
                        "net_pnl_quote",
                        "net_pnl",
                        "close_timestamp",
                        "executor_status",
                        "close_type",
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
                        "controller_name"]
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
            position_executor["close_timestamp"] = backtesting_executor["close_time"]
            position_executor["executor_status"] = None
            position_executor["close_type"] = backtesting_executor["close_type"]
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

            position_executor.set_index("timestamp", inplace=True)
            position_executor["datetime"] = pd.to_datetime(position_executor.index, unit="s")
            position_executor["close_datetime"] = pd.to_datetime(position_executor["close_timestamp"], unit="s")

            # Assigning an auto-increment 'id' column starting from 1
            position_executor["id"] = range(1, len(position_executor) + 1)

            # print("-last position_executor-----------")
            # print(position_executor.iloc[-1])

        return position_executor
        
    def get_backtesting_trade_fills(self, start_date=None, end_date=None):
        with self.session_maker() as session:
            query = self._get_backtesting_executors_query(start_date, end_date)
            try:
                backtesting_executor = pd.read_sql_query(text(query), session.connection())
            except Exception as e:
                print(e)
                return None
            
        # current binance rule
        contract_multiplier, maker_fee_rate, taker_fee_rate = self._get_contract_multiplier_fee(backtesting_executor)

        # open and close fills for each position executors
        all_trades = []
        for row in backtesting_executor.itertuples():
            if contract_multiplier == 1:
                open_base_amount = row.amount
                close_base_amount = row.amount
                open_notional_value = open_base_amount * row.price
                close_notional_value = close_base_amount * row.close_price
            else:
                # open_notional_value = row.amount * contract_multiplier
                # close_notional_value = row.amount * contract_multiplier
                open_base_amount = row.amount * contract_multiplier / row.price
                close_base_amount = row.amount * contract_multiplier / row.close_price
                open_notional_value = open_base_amount * row.close_price
                close_notional_value = close_base_amount * row.close_price

            open_trade = {
                "timestamp": row.timestamp,
                "strategy": row.strategy,
                "market": row.exchange,
                "symbol": row.symbol,
                "trade_type": row.side,
                "price": row.close,
                "amount": row.amount,
                "base_amount": open_base_amount,
                "quote_volume": open_notional_value,
                "net_amount": open_base_amount * row.side.upper().apply(lambda x: 1 if x == 'BUY' else -1),
                "net_amount_quote": open_notional_value * row.side.upper().apply(lambda x: 1 if x == 'BUY' else -1),
                "trade_fee": open_base_amount * taker_fee_rate, 
                "trade_fee_in_quote": open_notional_value * taker_fee_rate,
                # "trade_fee": (row.trade_pnl - row.net_pnl) / 2,
                # "trade_fee_in_quote": (row.trade_pnl - row.net_pnl) / 2 * (row.net_pnl_quote / row.net_pnl),
                "leverage": 1,
                "base_asset": row.symbol.split("-")[0],
                "quote_asset": row.symbol.split("-")[1],
                "order_id": None,
                "order_type": None,
                "position": None
            }
            close_trade = {
                "timestamp": row.close_time,
                "strategy": row.strategy,
                "market": row.exchange,
                "symbol": row.symbol,
                "trade_type": "SELL" if row.side == 'BUY' else "BUY",
                "price": row.close_price,
                "amount": row.amount,
                "base_amount": close_base_amount,
                "quote_volume": close_notional_value,
                "net_amount": close_base_amount * row.side.upper().apply(lambda x: 1 if x == 'BUY' else -1),
                "net_amount_quote": close_notional_value * row.side.upper().apply(lambda x: 1 if x == 'BUY' else -1),
                "trade_fee": close_base_amount * taker_fee_rate, 
                "trade_fee_in_quote": close_notional_value * taker_fee_rate,
                # "trade_fee": (row.trade_pnl - row.net_pnl) / 2,
                # "trade_fee_in_quote": (row.trade_pnl - row.net_pnl) / 2 * (row.net_pnl_quote / row.net_pnl),
                "leverage": 1,
                "base_asset": row.symbol.split("-")[0],
                "quote_asset": row.symbol.split("-")[1],
                "order_id": None,
                "order_type": None,
                "position": None
            }
            all_trades.append(open_trade)
            all_trades.append(close_trade)

        trade_fills = pd.DataFrame(all_trades)

        groupers = ["market", "symbol"]
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
        # trade_fills["timestamp"] = pd.to_datetime(trade_fills["timestamp"], unit="ms")
        # trade_fills["market"] = trade_fills["market"]
        # if amount is not of base_asset it's not true @luffy
        # trade_fills["quote_volume"] = trade_fills["price"] * trade_fills["amount"]

        # print("-last trade_fills-----------")
        # print(trade_fills.iloc[-1])

        return trade_fills
   

    