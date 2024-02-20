import os
# import streamlit as st

import pandas as pd
import numpy as np
from sqlalchemy import text #, create_engine
# from sqlalchemy.orm import sessionmaker
from datetime import timedelta #, datetime

# from utils.data_manipulation import StrategyData
from loopy_quant.loopy_data_manipulation import LoopyStrategyData
from loopy_quant.loopy_database_manager import LoopyDBManager


class LoopyRealDBManager(LoopyDBManager):
    # def __init__(self, db_name: str, executors_path: str = "data"):
    def __init__(self, db_name: str, start_date=None, end_date=None):

        super().__init__(db_name, start_date, end_date)
        self.strategy = 'loopy_macd_v1'.upper()

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
            if "real" in self.db_name:
                order_status = None
                market_data = load_data(self.get_market_data, self.start_date, self.end_date)
                orders, trade_fills, position_executors = load_data(self.get_trade_history_datum, self.start_date, self.end_date)
                strategy_data = LoopyStrategyData(orders, order_status, trade_fills, market_data, position_executors)
        
        return strategy_data
    
    @property
    def status(self):
        def load_data(table_loader, start_date=None, end_date=None):
            try:
                return table_loader(start_date, end_date)
            except Exception as e:
                print(e)
                return None  # Return None to indicate failure
            
        def get_data_status(data):
            return "Correct" if len(data) > 0 else "Error - No records matched"
            
        status = None
        if "postgres" in self.db_name:
            if "real" in self.db_name:
                market_data = load_data(self.get_market_data, self.start_date, self.end_date)
                orders, trade_fills, position_executors = load_data(self.get_trade_history_datum, self.start_date, self.end_date)
                status = {
                    "db_name": self.db_name,
                    "orders": get_data_status(orders),
                    "order_status": "Correct",
                    "trade_fill": get_data_status(trade_fills),
                    "market_data": get_data_status(market_data),
                    "position_executor": get_data_status(position_executors),
                }
      
        return status
    
    def _get_orders_group_by_id_legacy(self, order_log:pd.DataFrame) -> pd.DataFrame:
        # Group by 'exchange', 'symbol', 'id' and aggregate to find the max values for 'price' and 'amount'
        grouped_df = order_log.groupby(['exchange', 'symbol', 'id']).agg({
            'price': 'max',
            'amount': 'max'
        }).reset_index()

        # make sure sorted by 'timestamp' before next operations
        order_log = order_log.sort_values(by='timestamp')

        # Extract 'side', 'type', 'status' for first and last occurrence within each group
        creation_timestamp = order_log.drop_duplicates(subset=['exchange', 'symbol', 'id'], keep='first').set_index(['exchange', 'symbol', 'id'])[['timestamp']]
        last_update_timestamp = order_log.drop_duplicates(subset=['exchange', 'symbol', 'id'], keep='last').set_index(['exchange', 'symbol', 'id'])[['timestamp']]
        side = order_log.drop_duplicates(subset=['exchange', 'symbol', 'id'], keep='first').set_index(['exchange', 'symbol', 'id'])[['side']]
        type = order_log.drop_duplicates(subset=['exchange', 'symbol', 'id'], keep='last').set_index(['exchange', 'symbol', 'id'])[['type']]
        status = order_log.drop_duplicates(subset=['exchange', 'symbol', 'id'], keep='last').set_index(['exchange', 'symbol', 'id'])[['status']]

        # Combine additional info into a single DataFrame before joining
        additional_info = pd.concat([creation_timestamp, last_update_timestamp, side, type, status], axis=1)
        # Join this combined DataFrame to grouped_df
        grouped_df = grouped_df.join(additional_info, on=['exchange', 'symbol', 'id'])

        # Now, 'exchange', 'symbol', 'id' are columns and can be used for joining
        grouped_df['timestamp'] = grouped_df['last_update_timestamp']
        # grouped_df.set_index("timestamp", inplace=True)

        # Sort by 'id' as per the SQL 'ORDER BY'
        grouped_df = grouped_df.sort_values(by='id')
    
        return grouped_df
    
    # group by order_id @luffy
    def _get_orders_group_by_id(self, order_log:pd.DataFrame) -> pd.DataFrame:
        # Define custom aggregation dictionary
        aggregations = {
            'price': 'max',
            'amount': 'max',
            # 'timestamp': ['first', 'last']
            'timestamp': ['min', 'max']
        }

        # Group by 'exchange', 'symbol', 'id' and aggregate
        grouped_df = order_log.groupby(['exchange', 'symbol', 'id']).agg(aggregations)
        # Flatten MultiIndex columns
        grouped_df.columns = ['_'.join(col).strip() for col in grouped_df.columns.values]
        # Correct the columns renaming based on your aggregation
        # grouped_df.rename(columns={'price_max': 'price', 'amount_max': 'amount', 'timestamp_first': 'creation_timestamp', 'timestamp_last': 'last_update_timestamp'}, inplace=True)
        grouped_df.columns = ['price', 'amount', 'creation_timestamp', 'last_update_timestamp']

        # Reset index to flatten the DataFrame and make 'exchange', 'symbol', 'id' as columns
        grouped_df.reset_index(inplace=True)

        # Now, 'exchange', 'symbol', 'id' are columns and can be used for joining
        grouped_df['timestamp'] = grouped_df['last_update_timestamp']
        # grouped_df.set_index("timestamp", inplace=True)

        # Sort by 'id' as per the SQL 'ORDER BY'
        # grouped_df = grouped_df.sort_values(by='id')

        # make sure sorted by 'timestamp' before next operations
        order_log = order_log.sort_values(by='timestamp')

        # Extract 'side', 'type', 'status' for first and last occurrence within each group
        side = order_log.drop_duplicates(subset=['exchange', 'symbol', 'id'], keep='first').set_index(['exchange', 'symbol', 'id'])[['side']]
        type = order_log.drop_duplicates(subset=['exchange', 'symbol', 'id'], keep='last').set_index(['exchange', 'symbol', 'id'])[['type']]
        status = order_log.drop_duplicates(subset=['exchange', 'symbol', 'id'], keep='last').set_index(['exchange', 'symbol', 'id'])[['status']]
        # Convert column to uppercase
        side['side'] = side['side'].str.upper() 
        type['type'] = type['type'].str.upper() 
        status['status'] = status['status'].str.upper() 

        additional_info = pd.concat([side, type, status], axis=1)
        grouped_df = grouped_df.join(additional_info, on=['exchange', 'symbol', 'id'])
        
        # data validation testing @luffy 
        test_grouped_df = grouped_df[grouped_df['symbol'] == 'BTC-USD-PERP']
        return test_grouped_df
    
        # return grouped_df
    
    # join order and position table to extract executor infoes @luffy
    def _get_order_join_position(self, order_log:pd.DataFrame, pos_log:pd.DataFrame) -> pd.DataFrame:
        order_info = self._get_orders_group_by_id(order_log)

        # Step 1: Replicating 'order_info'
        # Sort the DataFrame by 'timestamp' and 'id' in descending order
        order_info = order_info.sort_values(by=['exchange', 'symbol', 'timestamp', 'id'], ascending=[True, True, True, False])
        order_info['cumcount'] = order_info.groupby(['exchange', 'symbol', 'timestamp']).cumcount()
        order_info['unique_timestamp'] = order_info.apply(lambda x: x['timestamp'] if x['cumcount'] == 0 else x['timestamp'] + pd.to_timedelta(x['cumcount'], unit='ms'), axis=1)
        # order_info['unique_timestamp'] = pd.to_timedelta(order_info['unique_timestamp'], unit='ms') + order_info['timestamp']

        # Drop the intermediate 'cumcount' column if it's no longer needed
        order_info.drop(columns=['cumcount'], inplace=True)

        # order_info['lag_side'] = order_info.sort_values(by=['creation_timestamp', 'last_update_timestamp', 'id']).groupby(['exchange', 'symbol'])['side'].shift(1)
        order_info = order_info[order_info['status'].isin(['FILLED', 'PARTIALLY_FILLED'])].copy()

        # Step 2: Replicating 'position_executor'
        # Prepare pos_info for merging
        pos_log['unique_timestamp'] = pos_log['timestamp']

        # Merge order_info with pos_info
        merged_df = pd.merge(order_info, pos_log, how='left', on=['exchange', 'symbol', 'unique_timestamp'], suffixes=('', '_pos'))

        # Apply conditional logic
        # merged_df['open'] = np.where(merged_df['price'] == merged_df['entry_price'], True, False)
        # merged_df['add'] = np.where((merged_df['entry_price'].isnull() | (merged_df['entry_price'] != merged_df['price'])) & (merged_df['side'] == merged_df['lag_side']), True, False)
        # merged_df['close'] = np.where((merged_df['amount_pos'].notnull() & (merged_df['amount_pos'] == 0)) | (merged_df['amount_pos'].isnull() & (merged_df['side'] != merged_df['lag_side'])), True, False)
        conditions = [
            (merged_df['amount_pos'].notnull()) & (merged_df['amount_pos'] == 0),
            (merged_df['amount_pos'].notnull()) & (np.abs(merged_df['amount_pos']) == np.abs(merged_df['amount']))
            # (merged_df['amount_pos'].notnull()) & (np.abs(merged_df['amount_pos']) != np.abs(merged_df['amount']))
        ]
        # Define choices corresponding to each condition
        # choices = ['close', 'open', 'add']
        choices = ['close', 'open']
        # Use numpy.select to apply conditions and choices, with '' as the default value
        merged_df['position'] = np.select(conditions, choices, default='')

        # Step 3: Final selection and ordering
        # final_df = merged_df[merged_df['symbol'] == 'BTC-USD-PERP'].sort_values(by=['creation_timestamp', 'last_update_timestamp', 'id'])
        final_df = merged_df.sort_values(by=['creation_timestamp', 'last_update_timestamp', 'id'])

        # data validation testing @luffy ------------------------------------------------------------------
        # test_order_df = order_info[order_info['symbol'] == 'BTC-USD-PERP'].sort_values(by=['creation_timestamp', 'last_update_timestamp', 'id'])
        # test_pos_log = pos_log[pos_log['symbol'] == 'BTC-USD-PERP'].sort_values(by='timestamp')
        # test_final_df = merged_df[merged_df['symbol'] == 'BTC-USD-PERP'].sort_values(by=['creation_timestamp', 'last_update_timestamp', 'id'])

        # print(f'------ data validation testing; length of order_group_id: {len(test_order_df)} ------------------')
        # print(test_order_df)
        # print(f'------ data validation testing; length of pos_log: {len(test_pos_log)} ------------------')
        # print(test_pos_log)
        # print(f'------ data validation testing; length of order_pos_join: {len(test_final_df)} ------------------')
        # print(test_final_df)
        # for index, row in test_final_df.iterrows():
        #     print(f'index of row: {index} ------------------')
        #     print(row)

        # return test_final_df
        # -------------------------------------------------------------------------------------------------

        return final_df
    
    def get_trade_history_datum(self, start_date=None, end_date=None):
        orders = None
        trade_fills = None
        position_executors = None

        with self.session_maker() as session:
            query = self._get_order_log_query(start_date, end_date)
            try:
                order_log = pd.read_sql_query(text(query), session.connection())
            except Exception as e:
                print(e)
                return None
            
            query = self._get_position_log_query(start_date, end_date)
            try:
                pos_log = pd.read_sql_query(text(query), session.connection())
            except Exception as e:
                print(e)
                return None
            
        orders = self.get_orders(order_log)
        order_join_pos_df = self._get_order_join_position(order_log, pos_log)
        trade_fills = self.get_trade_fills(order_join_pos_df)
        position_executors = self.get_position_executors(order_join_pos_df)

        print(f'------ trade stat; {start_date} ~ {end_date} ------------------')
        print(f'order count: {len(orders)} ------------------')
        print(f'trade count: {len(trade_fills)} ------------------')
        print(f'position count: {len(position_executors)} ------------------')

        return orders, trade_fills, position_executors
    
    def get_orders(self, order_log:pd.DataFrame) -> pd.DataFrame:

        grouped_df = self._get_orders_group_by_id(order_log)

        # current binance rule
        contract_multiplier, maker_fee_rate, taker_fee_rate = self._get_contract_multiplier_fee(grouped_df)

        all_orders = []
        for row in grouped_df.itertuples():
            if contract_multiplier == 1:
                notional_value = row.price * row.amount
                base_amount = row.amount
            else:
                notional_value = row.amount * contract_multiplier
                base_amount = row.amount * contract_multiplier / row.price
                
            order = {
                    "creation_timestamp": row.creation_timestamp,
                    "last_update_timestamp": row.last_update_timestamp,
                    # "strategy": self.strategy,
                    "market": row.exchange,
                    "symbol": row.symbol,
                    "id": row.id,
                    "trade_type": row.side.upper(),
                    "price": row.price,
                    "amount": row.amount,
                    "base_amount": base_amount,
                    "quote_volume": notional_value,
                    "base_asset": row.symbol.split("-")[0],
                    "quote_asset": row.symbol.split("-")[1],
                    "order_id": row.id,
                    "order_type": row.type.upper(),
                    "position": None,
                    "status": row.status
                }
            all_orders.append(order)

        orders = pd.DataFrame(all_orders)

        return orders

    def get_trade_fills_legacy(self, order_log:pd.DataFrame) -> pd.DataFrame:

        order_info = self._get_orders_group_by_id(order_log)

        # current binance rule
        contract_multiplier, maker_fee_rate, taker_fee_rate = self._get_contract_multiplier_fee(order_info)
            
        all_trades = []
        trade_info = order_info[order_info['status'].isin(['FILLED', 'PARTIALLY_FILLED'])].copy()

        for row in trade_info.itertuples():
            order_type = row.type.upper()
            if contract_multiplier == 1:
                notional_value = row.price * row.amount
                base_amount = row.amount
            else:
                notional_value = row.amount * contract_multiplier
                base_amount = row.amount * contract_multiplier / row.price
                
            trade = {
                    "timestamp": row.last_update_timestamp,
                    "strategy": self.strategy,
                    "market": row.exchange,
                    "symbol": row.symbol,
                    "trade_type": row.side.upper(),
                    "price": row.price,
                    "amount": row.amount,
                    "base_amount": base_amount,
                    "quote_volume": notional_value,
                    "trade_fee": base_amount * taker_fee_rate if order_type == 'MARKET' else base_amount * maker_fee_rate, 
                    "trade_fee_in_quote": notional_value * taker_fee_rate if order_type == 'MARKET' else notional_value * maker_fee_rate,
                    "leverage": 1,
                    "base_asset": row.symbol.split("-")[0],
                    "quote_asset": row.symbol.split("-")[1],
                    "order_id": row.id,
                    "order_type": order_type,
                    "position": None,
                    "status": row.status
                }
            all_trades.append(trade)

        trade_fills = pd.DataFrame(all_trades)

        groupers = ["market", "symbol"]
        trade_fills["cum_fees_in_quote"] = trade_fills.groupby(groupers)["trade_fee_in_quote"].cumsum()
        # trade_fills["net_amount"] = trade_fills['amount'] * trade_fills['trade_type'].apply(lambda x: 1 if x == 'BUY' else -1)
        trade_fills["net_amount"] = trade_fills['base_amount'] * trade_fills['trade_type'].apply(lambda x: 1 if x == 'BUY' else -1)
        trade_fills["net_amount_quote"] = trade_fills['net_amount'] * trade_fills['price']
        trade_fills["cum_net_amount"] = trade_fills.groupby(groupers)["net_amount"].cumsum()
        trade_fills["unrealized_trade_pnl"] = -1 * trade_fills.groupby(groupers)["net_amount_quote"].cumsum()
        trade_fills["inventory_cost"] = trade_fills["cum_net_amount"] * trade_fills["price"]
        trade_fills["realized_trade_pnl"] = trade_fills["unrealized_trade_pnl"] + trade_fills["inventory_cost"]
        trade_fills["net_realized_pnl"] = trade_fills["realized_trade_pnl"] - trade_fills["cum_fees_in_quote"]
        trade_fills["realized_pnl"] = trade_fills.groupby(groupers)["net_realized_pnl"].diff()
        trade_fills["gross_pnl"] = trade_fills.groupby(groupers)["realized_trade_pnl"].diff()
        trade_fills["trade_fee"] = trade_fills.groupby(groupers)["cum_fees_in_quote"].diff()

        return trade_fills
    
    def get_trade_fills(self, order_join_pos_df:pd.DataFrame) -> pd.DataFrame:
        # current binance rule
        contract_multiplier, maker_fee_rate, taker_fee_rate = self._get_contract_multiplier_fee(order_join_pos_df)
        
        # Initialize a list to track indices of 'open'/'add' rows until 'close' is encountered
        position_trades = []
        position_size = 0
        position_side = None
        position_status = 'open'

        trade_fills = pd.DataFrame()

        for index, row in order_join_pos_df.iterrows():
            # chekc last index @luffy
            # if index == order_join_pos_df.index[-1]:
            #     print('now!!!!!')

            # check position status (open/close)
            if row['position'] == 'close':
                if position_side is not None and position_side != row.side:
                    # check closing size
                    if position_size == row.amount:
                        position_status = 'close'
                    else:
                        position_status = None
                else:
                    position_status = None
            elif row['position'] == '':
                if len(position_trades) > 0:
                    if position_side is not None and position_side != row.side:
                        # check closing size
                        if position_size == row.amount:
                            position_status = 'close'
                        else:
                            position_status = None

            if position_status == 'open':
                order_type = row.type.upper()
                if contract_multiplier == 1:
                    base_amount = row.amount
                    notional_value = base_amount * row.price
                else:
                    base_amount = row.amount * contract_multiplier / row.price
                    # notional_value = None
                    notional_value = row.amount * contract_multiplier

                open_trade = {
                    "timestamp": row.timestamp,
                    "strategy": self.strategy,
                    "market": row.exchange,
                    "symbol": row.symbol,
                    "trade_type": row.side.upper(),
                    "price": row.price,
                    "amount": row.amount,
                    "base_amount": base_amount,
                    "quote_volume": notional_value,
                    # "net_amount": base_amount * row.side.upper().apply(lambda x: 1 if x == 'BUY' else -1),
                    # "net_amount_quote": notional_value * row.side.upper().apply(lambda x: 1 if x == 'BUY' else -1) if notional_value is not None else None,
                    "net_amount": base_amount * np.where(row.side == 'BUY', 1, -1),
                    "net_amount_quote": notional_value * np.where(row.side == 'BUY', 1, -1), #if notional_value is not None else None,
                    "trade_fee": base_amount * taker_fee_rate if order_type == 'MARKET' else base_amount * maker_fee_rate, 
                    "trade_fee_in_quote": notional_value * taker_fee_rate if order_type == 'MARKET' else base_amount * maker_fee_rate, 
                    # "trade_fee_in_quote": base_amount * row.price * taker_fee_rate if order_type == 'MARKET' else base_amount * maker_fee_rate,
                    "leverage": 1,
                    "level": len(position_trades),
                    "base_asset": row.symbol.split("-")[0],
                    "quote_asset": row.symbol.split("-")[1],
                    "order_id": row.id,
                    "order_type": order_type,
                    "position": None
                }

                position_trades.append(open_trade)

                position_size = position_size + row.amount 
                position_side = row.side
            elif position_status == 'close':
                order_type = row.type.upper()
                if contract_multiplier == 1:
                    base_amount = row.amount
                    notional_value = base_amount * row.price
                else:
                    base_amount = row.amount * contract_multiplier / row.price
                    # notional_value = base_amount * row.price
                    notional_value = row.amount * contract_multiplier

                close_trade = {
                    "timestamp": row.timestamp,
                    "strategy": self.strategy,
                    "market": row.exchange,
                    "symbol": row.symbol,
                    "trade_type": row.side.upper(),
                    "price": row.price,
                    "amount": row.amount,
                    "base_amount": base_amount,
                    "quote_volume": notional_value,
                    "net_amount": base_amount * np.where(row.side == 'BUY', 1, -1),
                    "net_amount_quote": notional_value * np.where(row.side == 'BUY', 1, -1),
                    "trade_fee": base_amount * taker_fee_rate if order_type == 'MARKET' else base_amount * maker_fee_rate, 
                    "trade_fee_in_quote": notional_value * taker_fee_rate if order_type == 'MARKET' else base_amount * maker_fee_rate, 
                    "leverage": 1,
                    "level": -1,
                    "base_asset": row.symbol.split("-")[0],
                    "quote_asset": row.symbol.split("-")[1],
                    "order_id": row.id,
                    "order_type": order_type,
                    "position": None
                }
                
                # For 'close', update the 'net_amount_quote' for all tracked 'open'/'add' rows
                # for trade in position_trades:
                #     if trade['quote_volume'] is None:
                #         trade['quote_volume'] = trade['base_amount'] * row.price
                #         trade['net_amount_quote'] = trade['net_amount'] * row.price

                # update trade_fills
                position_trades.append(close_trade)
                # trade_fills.append(position_trades)
                trade_fills = pd.concat([trade_fills, pd.DataFrame(position_trades)], ignore_index=True)

                # Reset the list for the next 'close'
                position_trades = []
                # open_add_indices = []
                position_size = 0
                position_side = None
                position_status = 'open'
            # elif position_status is None:
            else:
                position_status = 'open'
                print('-----------Warning! position status checking problem!-------------------')
                print(f'position: {row.position}')
                print(f'position_side: {position_side}, position_size: {position_size}')
                print(f'order_side: {row.side}, order_size: {row.amount}')
                print(row)

        # add not closed positions remained
        if len(position_trades) > 0:
            trade_fills = pd.concat([trade_fills, pd.DataFrame(position_trades)], ignore_index=True)
            position_trades = []
            position_size = 0
            position_side = None

        # print('----------- last trade_fill -------------------')
        # print(trade_fills.iloc[-1])
        print(f'-------------- before pnl calculation; trade_fills count: {len(trade_fills)} ------------------------')
        before_trade_fills = trade_fills[['timestamp', 
                                        #   'market', 
                                          'symbol', 
                                          'order_id',
                                          'trade_type', 
                                          'price', 
                                          'amount', 
                                          'net_amount',
                                          'net_amount_quote'
                                        ]]
        print(before_trade_fills)
        
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

        print(f'-------------- after pnl calculation; trade_fills count: {len(trade_fills)} ------------------------')
        pnl_trade_fills = trade_fills[['timestamp', 
                                        #   'market', 
                                          'symbol', 
                                          'order_id',
                                          'trade_type', 
                                          'unrealized_trade_pnl', 
                                          'inventory_cost', 
                                          'realized_trade_pnl',
                                          'net_realized_pnl'
                                        ]]
        print(pnl_trade_fills)

        return trade_fills

    def get_position_executors(self, order_join_pos_df:pd.DataFrame) -> pd.DataFrame:
        # join order and position table to extract executor infoes @luffy
        # join_df = self._get_order_join_position(order_log, pos_log)
        
        # Initialize an empty DataFrame to store the result
        # position_executors = pd.DataFrame()
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
            "controller_name",
            "id",
            "level",
            # extra @luffy
            "base_amount",
            "trade_fee",
            "trade_fee_in_quote"
        ]
        position_executors = pd.DataFrame(columns=columns)
        
        # current binance rule
        contract_multiplier, maker_fee_rate, taker_fee_rate = self._get_contract_multiplier_fee(order_join_pos_df)
        # Initialize a list to track indices of 'open'/'add' rows until 'close' is encountered
        open_add_indices = []
        position_size = 0
        position_side = None
        position_status = 'open'

        for index, row in order_join_pos_df.iterrows():
            # check position status (open/close)
            if row['position'] == 'close':
                if position_side is not None and position_side != row.side:
                    # check closing size
                    if position_size == row.amount:
                        position_status = 'close'
                    else:
                        position_status = None
                else:
                    position_status = None
            elif row['position'] == '':
                if len(open_add_indices) > 0:
                    if position_side is not None and position_side != row.side:
                        # check closing size
                        if position_size == row.amount:
                            position_status = 'close'
                        else:
                            position_status = None

            if position_status == 'open':
            # if row['position'] in ['open', 'add']:
                # if contract_multiplier == 1:
                #     base_amount = row.amount
                #     notional_value = base_amount * row.price
                # else:
                #     base_amount = row.amount * contract_multiplier / row.price
                #     notional_value = base_amount * row.price
                    # notional_value = row.amount * contract_multiplier

                if contract_multiplier == 1:
                    base_amount = row.amount
                    # notional_value = base_amount * row.price
                else:
                    base_amount = row.amount * contract_multiplier / row.price
                    # notional_value = None

                executor = {
                    "timestamp": row.timestamp,
                    "close_timestamp": None,
                    "controller_name": self.strategy,
                    "exchange": row.exchange,
                    "trading_pair": row.symbol,
                    "side": row.side.upper(),
                    "entry_price": row.price,
                    "close_price": None,
                    "amount": row.amount,
                    "base_amount": base_amount,
                    # "notional_value_in_quote": notional_value,
                    # "quote_volume": notional_value,
                    "trade_fee": base_amount * (taker_fee_rate + maker_fee_rate), 
                    "trade_fee_in_quote": base_amount * row.price * maker_fee_rate, 
                    "leverage": 1,
                    "id": row.id,
                    "open_order_type": row.type.upper(),
                    "level": len(open_add_indices)
                }
                # Duplicate the row in the result DataFrame
                # position_executors = position_executors.append(executor, ignore_index=True)
                new_index = len(position_executors)
                position_executors.loc[new_index] = executor
                # Keep track of the index of this 'open'/'add' row
                open_add_indices.append(new_index)
                position_size = position_size + row.amount 
                position_side = row.side
            elif position_status == 'close':
                # For 'close', update the 'close_timestamp' for all tracked 'open'/'add' rows
                for open_add_index in open_add_indices:
                    position_executors.at[open_add_index, 'close_timestamp'] = row.timestamp
                    position_executors.at[open_add_index, 'close_price'] = row.price
                    position_executors.at[open_add_index, 'close_order_type'] = row.type.upper()

                    base_amount = position_executors.at[open_add_index, 'base_amount']
                    entry_price = position_executors.at[open_add_index, 'entry_price']
                    close_price = position_executors.at[open_add_index, 'close_price']
                    side = position_executors.at[open_add_index, 'side']

                    if contract_multiplier == 1:
                        # spot or vanilla contract
                        # position_executors.at[open_add_index, 'trade_pnl_quote'] = base_amount * (close_price - entry_price) * side.apply(lambda x: 1 if x == 'BUY' else -1)
                        position_executors.at[open_add_index, 'trade_pnl_quote'] = base_amount * (close_price - entry_price) * np.where(side == 'BUY', 1, -1)
                        position_executors.at[open_add_index, 'trade_fee_in_quote'] += base_amount * row.price * taker_fee_rate,
                    else:
                        # inverse contract
                        # close_base_amount = row.amount * contract_multiplier / row.price
                        close_base_amount = position_executors.at[open_add_index, 'amount'] * contract_multiplier / row.price
                        # position_executors.at[open_add_index, 'trade_pnl_quote'] = (base_amount - close_base_amount) * row.price * side.apply(lambda x: 1 if x == 'BUY' else -1)
                        position_executors.at[open_add_index, 'trade_pnl_quote'] = (base_amount - close_base_amount) * row.price * np.where(side == 'BUY', 1, -1)    
                        position_executors.at[open_add_index, 'trade_fee_in_quote'] += close_base_amount * row.price * taker_fee_rate,  
                        # print('-----------trade_pnl_quote_chekc!!-------------------')
                        # print(f'order_id: {position_executors.at[open_add_index, "id"]}')
                        # print(f'base_amount: {base_amount}')
                        # print(f'close_base_amount: {close_base_amount}')
                        # print(f'close_price: {row.price}')
                        # trade_pnl_quote = (base_amount - close_base_amount) * row.price * np.where(side == 'BUY', 1, -1)
                        # print(f'trade_pnl_quote: {trade_pnl_quote}')  
                        # print('------------------------------------------------------')          
                    
                    # position_executors.at[open_add_index, 'trade_fee_in_quote'] += base_amount * row.price * taker_fee_rate,
                    position_executors.at[open_add_index, 'net_pnl_quote'] = position_executors.at[open_add_index, 'trade_pnl_quote'] - position_executors.at[open_add_index, 'trade_fee_in_quote']
                   
                # Reset the list for the next 'close'
                open_add_indices = []
                position_size = 0
                position_side = None
                position_status = 'open'
            # elif position_status is None:
            else:
                position_status = 'open'
                print('-----------Warning! position status checking problem!-------------------')
                print(row)

        groupers = ["exchange", "trading_pair"]
        # position_executors["trade_pnl_quote"] = position_executors.groupby(groupers)["trade_pnl_quote"].cumsum()
        position_executors["cum_fee_quote"] = position_executors.groupby(groupers)["trade_fee_in_quote"].cumsum()
        # position_executors["net_pnl_quote"] = position_executors.groupby(groupers)["net_pnl_quote"].cumsum()
        position_executors["trade_pnl"] = None
        position_executors["net_pnl"] = None
        position_executors["executor_status"] = None
        position_executors["close_type"] = None
        # temporary for visualizing
        position_executors["sl"] = 1.0
        position_executors["tp"] = 1.0
        position_executors["tl"] = position_executors["timestamp"] + timedelta(days=1)
        # position_executors["open_order_type"] = 'LIMIT'
        position_executors["take_profit_order_type"] = position_executors['close_order_type']
        position_executors["stop_loss_order_type"] = position_executors['close_order_type']
        position_executors["time_limit_order_type"] = position_executors['close_order_type']

        # extra
        position_executors["datetime"] =position_executors["timestamp"]
        position_executors["close_datetime"] = position_executors["close_timestamp"]
        # for performance_candles of data_viz @luffy
        position_executors["side"] = position_executors["side"].apply(lambda x: 1 if x == 'BUY' else -1)

        position_executors.set_index("timestamp", inplace=True)

        # print(f'-------------- position executors; executors count: {len(position_executors)} ------------------------')
        # print(f'-------------- last position executor ----------------------------------------------------------------')
        # print(position_executors.iloc[-1])

        # position_executors["datetime"] = pd.to_datetime(position_executors.index, unit="s")
        # position_executors["datetime"] = pd.to_datetime(position_executors["timestamp"], unit="s")
        # position_executors["close_datetime"] = pd.to_datetime(position_executors["close_timestamp"], unit="s")

        print(f'-------------- position executors; executors count: {len(position_executors)} ------------------------')
        # print(position_executors)
        # for index, row in position_executors.iterrows():
        #     print(f'-------------- each position executors; index of row: {index} ------------------')
        #     print(row)
        summary = position_executors[[
                                    #   'datetime', 
                                      'close_datetime',
                                    #   'exchange', 
                                      'trading_pair',
                                      'side',
                                      'entry_price',
                                      'close_price',
                                      'amount',
                                      'base_amount',
                                      'trade_pnl_quote',
                                      'net_pnl_quote',
                                      'id'
                                    ]]
        print(summary)

        return position_executors

    
    
    