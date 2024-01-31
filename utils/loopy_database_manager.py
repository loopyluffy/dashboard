import os
import streamlit as st

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from utils.data_manipulation import StrategyData
from utils.loopy_data_manipulation import LoopyBacktestingStrategyData

from utils.database_manager import DatabaseManager


def get_bots_data_paths():
    root_directory = "hummingbot_files/bots"
    bots_data_paths = {"General / Uploaded data": "data"}
    reserved_word = "hummingbot-"
    # Walk through the directory tree
    for dirpath, dirnames, filenames in os.walk(root_directory):
        for dirname in dirnames:
            if dirname == "data":
                parent_folder = os.path.basename(dirpath)
                if parent_folder.startswith(reserved_word):
                    bots_data_paths[parent_folder] = os.path.join(dirpath, dirname)
            if "dashboard" in bots_data_paths:
                del bots_data_paths["dashboard"]
    data_sources = {key: value for key, value in bots_data_paths.items() if value is not None}
    return data_sources


def get_databases():
    databases = {}
    bots_data_paths = get_bots_data_paths()
    for source_name, source_path in bots_data_paths.items():
        sqlite_files = {}
        for db_name in os.listdir(source_path):
            if db_name.endswith(".sqlite"):
                sqlite_files[db_name] = os.path.join(source_path, db_name)
        databases[source_name] = sqlite_files

    # external database added in list @luffy
    # db_name = os.environ['DB_NAME']
    # databases["External / Databases"] = {"Postgresql": db_name}
    databases["External / Databases"] = {"Postgresql(Real)": "postgres_real", "Postgresql(Backtesting)": "postgres_backtesting"}

    if len(databases) > 0:
        return {key: value for key, value in databases.items() if value}
    else:
        return None


class LoopyDBManager(DatabaseManager):
    def __init__(self, db_name: str, executors_path: str = "data"):
        self.db_name = db_name
        # TODO: Create db path for all types of db
        if "sqlite" in db_name:
            self.db_path = f'sqlite:///{os.path.join(db_name)}'
            self.engine = create_engine(self.db_path, connect_args={'check_same_thread': False})
        # add db path for postgresql @luffy
        elif "postgres" in db_name:
            if "backtesting" in db_name:
                try:
                    db_engine = "postgresql+psycopg2"
                    db_host = os.environ['DB_PATH']
                    db_name = os.environ['DB_NAME']
                    db_username = os.environ['DB_USERNAME']
                    db_password = os.environ['DB_PASSWORD']
                    db_port = 5432
                    db_path = f"{db_engine}://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"
                    self.db_path = db_path
                    self.engine = create_engine(self.db_path)
                except Exception as e:
                    print(e)
            # if "real" in db_name:
            else:
                self.engine = None
                return
        else:
            self.engine = None
            return
        
        # self.engine = create_engine(self.db_path, connect_args={'check_same_thread': False})
        # f"{self.db_engine}://{self.db_username}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        # db_engine: postgresql+psycopg2
        self.session_maker = sessionmaker(bind=self.engine)

    def get_strategy_data(self, config_file_path=None, start_date=None, end_date=None):
        def load_data(table_loader):
            try:
                return table_loader()
            except Exception as e:
                print(e)
                return None  # Return None to indicate failure

        # Use load_data to load tables
        # different query route on source type @luffy
        if "sqlite" in self.db_name:
            orders = load_data(self.get_orders)
            trade_fills = load_data(self.get_trade_fills)
            order_status = load_data(self.get_order_status)
            market_data = load_data(self.get_market_data)
            position_executor = load_data(self.get_position_executor_data)
        elif "postgres" in self.db_name:
            if "backtesting" in self.db_name:
                market_data = load_data(self.get_backtesting_market_data)
                trade_fills = load_data(self.get_backtesting_trade_fills)
                position_executor = load_data(self.get_backtesting_executor_data)

                return LoopyBacktestingStrategyData(trade_fills, market_data, position_executor)
            # if "real" in db_name:
            else:
                return None
        else:
            return None

        strategy_data = StrategyData(orders, order_status, trade_fills, market_data, position_executor)
        return strategy_data
    
    @staticmethod
    def _get_table_status(table_loader):
        try:
            data = table_loader()
            return "Correct" if len(data) > 0 else f"Error - No records matched"
        except Exception as e:
            return f"Error - {str(e)}"

    @property
    def status(self):
        status = None
        if "sqlite" in self.db_name:
            status = {
                "db_name": self.db_name,
                "trade_fill": self._get_table_status(self.get_trade_fills),
                "orders": self._get_table_status(self.get_orders),
                "order_status": self._get_table_status(self.get_order_status),
                "market_data": self._get_table_status(self.get_market_data),
                "position_executor": self._get_table_status(self.get_position_executor_data),
            }
        elif "postgres" in self.db_name:
            if "backtesting" in self.db_name:
                status = {
                    "db_name": self.db_name,
                    "orders": "Correct",
                    "order_status": "Correct",
                    "trade_fill": self._get_table_status(self.get_backtesting_trade_fills),
                    "market_data": self._get_table_status(self.get_backtesting_market_data),
                    "position_executor": self._get_table_status(self.get_backtesting_executor_data),
                }
      
        return status

    @staticmethod
    def _get_market_info_query(start_date=None, end_date=None):
        # assign time duration forcely to test a external connection @luffy
        # ------------------------------------------
        start_date = '2023-02-01'
        end_date = '2023-02-10'
        # start_date = pd.Timestamp(start_date)
        # start_date = start_date.timestamp() # convert to unix timestamp in seconds
        # end_date = pd.Timestamp(end_date)
        # end_date = end_date.timestamp() # convert to unix timestamp in seconds
        # ------------------------------------------
        
        query = "SELECT * FROM market_info_log"
        conditions = []
        # if start_date:
        #     conditions.append(f"timestamp >= '{start_date * 1e6}'")
        # if end_date:
        #     conditions.append(f"timestamp <= '{end_date * 1e6}'")
        if start_date:
            conditions.append(f"timestamp >= '{start_date}'")
        if end_date:
            conditions.append(f"timestamp <= '{end_date}'")
        if conditions:
            query += f" WHERE {' AND '.join(conditions)}"
        return query

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

    def get_backtesting_market_data(self, start_date=None, end_date=None):
        with self.session_maker() as session:
            # query = self._get_market_data_query(start_date, end_date)
            query = self._get_market_info_query(start_date, end_date)
            try:
                market_info = pd.read_sql_query(text(query), session.connection())
            except Exception as e:
                print(e)
                return None
            
            columns = ["timestamp", "exchange", "trading_pair", "mid_price", "best_bid", "best_ask", "order_book", "trade_price"]
            market_data = pd.DataFrame(columns=columns)
            # market_data["timestamp"] = pd.to_datetime(market_data["timestamp"] / 1e6, unit="ms")
            # market_data.set_index("timestamp", inplace=True)
            # market_data["mid_price"] = market_data["mid_price"] / 1e6
            # market_data["best_bid"] = market_data["best_bid"] / 1e6
            # market_data["best_ask"] = market_data["best_ask"] / 1e6
            market_data["timestamp"] = market_info["timestamp"]
            market_data["exchange"] = market_info["exchange"]
            market_data["trading_pair"] = market_info["symbol"]
            market_data["best_bid"] = market_info["ticker_bid"]
            market_data["best_ask"] = market_info["ticker_ask"]
            market_data["mid_price"] = (market_info["ticker_bid"] + market_info["ticker_ask"]) / 2
            market_data["trade_price"] = market_info["trade_price"]

            market_data.set_index("timestamp", inplace=True)

            # print("-last market_data-----------")
            # print(market_data.iloc[-1])

        return market_data 
    
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

            print("-last position_executor-----------")
            print(position_executor.iloc[-1])

        return position_executor
        
    def get_backtesting_trade_fills(self, start_date=None, end_date=None):
        with self.session_maker() as session:
            query = self._get_backtesting_executors_query(start_date, end_date)
            try:
                backtesting_executor = pd.read_sql_query(text(query), session.connection())
            except Exception as e:
                print(e)
                return None
            
        # columns = [
        #         # "exchange_trade_id",  # Keep the key attribute first in the list
        #         # "config_file_path",
        #         "strategy",
        #         "market",
        #         "symbol",
        #         "base_asset",
        #         "quote_asset",
        #         "timestamp",
        #         "order_id",
        #         "trade_type",
        #         "order_type",
        #         "price",
        #         "amount",
        #         "leverage",
        #         "trade_fee",
        #         "trade_fee_in_quote",
        #         "position"
        #     ]
        # trade_fills = pd.DataFrame(columns=columns)

        # open and close fills for each position executors
        all_trades = []
        for row in backtesting_executor.itertuples():
            open_trade = {
                "timestamp": row.timestamp,
                "strategy": row.strategy,
                "market": row.exchange,
                "symbol": row.symbol,
                "trade_type": row.side,
                "price": row.close,
                "amount": row.amount,
                "trade_fee": (row.trade_pnl - row.net_pnl) / 2,
                "trade_fee_in_quote": (row.trade_pnl - row.net_pnl) / 2 * (row.net_pnl_quote / row.net_pnl),
                "leverage": 1,
                "base_asset": None,
                "quote_asset": None,
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
                "trade_fee": (row.trade_pnl - row.net_pnl) / 2,
                "trade_fee_in_quote": (row.trade_pnl - row.net_pnl) / 2 * (row.net_pnl_quote / row.net_pnl),
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
        trade_fills["net_amount"] = trade_fills['amount'] * trade_fills['trade_type'].apply(lambda x: 1 if x == 'BUY' else -1)
        trade_fills["net_amount_quote"] = trade_fills['net_amount'] * trade_fills['price']
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
        trade_fills["quote_volume"] = trade_fills["price"] * trade_fills["amount"]

        print("-last trade_fills-----------")
        print(trade_fills.iloc[-1])

        return trade_fills
   

    