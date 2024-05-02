import os
import streamlit as st

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# from utils.data_manipulation import StrategyData
# from loopy_quant.loopy_data_manipulation import LoopyBacktestingStrategyData

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


def get_databases(sqlite=False, backtesting=False):
    databases = {}
    if sqlite == True:
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
    # databases["External / Databases"] = {"Postgresql(Real)": "postgres_real", "Postgresql(Backtesting)": "postgres_backtesting"}
    if backtesting == False:
        databases["External / Databases"] = {"Postgresql(Real)": "postgres_real"}
    else:
        databases["External / Databases"] = {"Postgresql(Backtesting)": "postgres_backtesting"}

    if len(databases) > 0:
        return {key: value for key, value in databases.items() if value}
    else:
        return None


class LoopyDBManager(DatabaseManager):
    # def __init__(self, db_name: str, executors_path: str = "data"):
    def __init__(self, db_name: str, start_date=None, end_date=None):
        self.db_name = db_name
        self.start_date = start_date
        self.end_date = end_date
        self.engine = None
        self.session_maker = None

        self.candle_engine = None
        self.candle_session_maker = None

        if "postgres" in db_name:
            if "backtesting" in db_name:
                db_host = os.environ['DB_BACKTESTING_PATH']
            elif "real" in db_name:
                db_host = os.environ['DB_REAL_PATH']
            else:
                db_host = os.environ['DB_PATH']
            
            try:
                db_engine = "postgresql+psycopg2"
                # db_host = os.environ['DB_PATH']
                db_name = os.environ['DB_NAME']
                db_username = os.environ['DB_USERNAME']
                db_password = os.environ['DB_PASSWORD']
                db_port = 5432
                db_path = f"{db_engine}://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"
                self.db_path = db_path
                self.engine = create_engine(self.db_path)
                self.session_maker = sessionmaker(bind=self.engine)
            except Exception as e:
                print(e)

        if self.session_maker is not None:
            try:
                db_candle_host = os.environ['DB_CANDLE_PATH']
                db_candle_path = f"{db_engine}://{db_username}:{db_password}@{db_candle_host}:{db_port}/{db_name}"
                self.db_candle_path = db_candle_path
                self.candle_engine = create_engine(db_candle_path)
                self.candle_session_maker = sessionmaker(bind=self.candle_engine)
            except Exception as e:
                print(e)

    @staticmethod
    def _get_candle_info_query(exchange=None, trading_pair=None, start_date=None, end_date=None):
        # assign time duration forcely to test a external connection @luffy
        # ------------------------------------------
        # start_date = '2023-02-01'
        # end_date = '2023-02-10'
        # start_date = pd.Timestamp(start_date)
        # start_date = start_date.timestamp() # convert to unix timestamp in seconds
        # end_date = pd.Timestamp(end_date)
        # end_date = end_date.timestamp() # convert to unix timestamp in seconds
        # ------------------------------------------
        
        query = "SELECT * FROM view_5m_candle"
        conditions = []
        # if start_date:
        #     conditions.append(f"timestamp >= '{start_date * 1e6}'")
        # if end_date:
        #     conditions.append(f"timestamp <= '{end_date * 1e6}'")
        if exchange:
            conditions.append(f"exchange = '{exchange}'")
        if trading_pair:
            conditions.append(f"symbol = '{trading_pair}'")
        if start_date:
            conditions.append(f"date >= '{start_date}'")
        if end_date:
            conditions.append(f"date <= '{end_date}'")
        if conditions:
            query += f" WHERE {' AND '.join(conditions)}"
        return query
    
    @staticmethod
    def _get_market_info_query(start_date=None, end_date=None):
        # assign time duration forcely to test a external connection @luffy
        # ------------------------------------------
        # start_date = '2023-02-01'
        # end_date = '2023-02-10'
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
    def _get_order_log_query(start_date=None, end_date=None):
        query = "SELECT * FROM order_log"
        conditions = []
        if start_date:
            conditions.append(f"timestamp >= '{start_date}'")
        if end_date:
            conditions.append(f"timestamp <= '{end_date}'")
        if conditions:
            query += f" WHERE {' AND '.join(conditions)}"
        return query
    
    @staticmethod
    def _get_position_log_query(start_date=None, end_date=None):
        query = "SELECT * FROM position_log_01"
        conditions = []
        if start_date:
            conditions.append(f"timestamp >= '{start_date}'")
        if end_date:
            conditions.append(f"timestamp <= '{end_date}'")
        if conditions:
            query += f" WHERE {' AND '.join(conditions)}"
        return query
    
    # return contract multiplier, maker/taker fee rate 
    # temporary codes for perpetual inverse @luffy
    @staticmethod
    def _get_contract_multiplier_fee(df: pd.DataFrame):
        last_row = df.iloc[-1]
        
        contract_multiplier = 1
        maker_fee_rate = 0.0002
        taker_fee_rate = 0.0005
        
        if "binance_delivery" == last_row.exchange.lower():
            if "btc-usd-perp" in last_row.symbol.lower():
                contract_multiplier = 100
            else:
                contract_multiplier = 10

        return contract_multiplier, maker_fee_rate, taker_fee_rate
    
    @staticmethod
    def _get_contract_multiplier_fee_for_each(row: pd.Series):
        if row is None:
            return None, None, None
        
        contract_multiplier = 1
        maker_fee_rate = 0.0002
        taker_fee_rate = 0.0005
        
        if "binance_delivery" == row.exchange.lower():
            if "btc-usd-perp" in row.symbol.lower():
                contract_multiplier = 100
            else:
                contract_multiplier = 10

        return contract_multiplier, maker_fee_rate, taker_fee_rate
    
    def get_candle_data(self, exchange=None, trading_pair=None, start_date=None, end_date=None):
        with self.candle_session_maker() as session:
            query = self._get_candle_info_query(exchange=exchange, trading_pair=trading_pair, start_date=start_date, end_date=end_date)
            try:
                candle_info = pd.read_sql_query(text(query), session.connection())
            except Exception as e:
                print(e)
                return None
            
            columns = ['timestamp','exchange','trading_pair','open','high','low','close','volume','quote_asset_volume']
            candle_data = pd.DataFrame(columns=columns)
            # market_data["timestamp"] = pd.to_datetime(market_data["timestamp"] / 1e6, unit="ms")
            candle_data["timestamp"] = candle_info["date"]
            candle_data["exchange"] = candle_info["exchange"]
            candle_data["trading_pair"] = candle_info["symbol"]
            candle_data["open"] = candle_info["open_price"]
            candle_data["high"] = candle_info["high_price"]
            candle_data["low"] = candle_info["low_price"]
            candle_data["close"] = candle_info["close_price"]
            candle_data["volume"] = candle_info["volume"]
            candle_data["quote_asset_volumne"] = candle_info["value"]

            # candle_data.set_index("timestamp", inplace=True)
            candle_data.set_index("timestamp")

            # print("-last candle_data-----------")
            # print(candle_data.tail(1))
            # print(candle_data.iloc[0])
            # print(candle_data.iloc[-1]["timestamp"])

        return candle_data 

    def get_market_data(self, start_date=None, end_date=None):
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
                # market_data["mid_price"] = (market_info["ticker_bid"] + market_info["ticker_ask"]) / 2
                # market_data["trade_price"] = market_info["trade_price"]
                market_data["mid_price"] = market_info["trade_price"]

                market_data.set_index("timestamp", inplace=True)

                print("-last market_data-----------")
                print(market_data.iloc[-1])
                # print(market_data.iloc[-1]["timestamp"])

            return market_data 