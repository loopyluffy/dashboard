import datetime
# from dataclasses import dataclass
import pandas as pd
import numpy as np

from utils.data_manipulation import StrategyData, SingleMarketStrategyData


class LoopyStrategyData(StrategyData):
    # orders: pd.DataFrame
    # order_status: pd.DataFrame
    # trade_fill: pd.DataFrame
    # market_data: pd.DataFrame = None
    # position_executor: pd.DataFrame = None

    # def __init__(self, trade_fill: pd.DataFrame, market_data: pd.DataFrame, position_executor: pd.DataFrame):
    #     super().__init__(None, None, trade_fill, market_data, position_executor)

    # @property
    # def strategy_summary(self):
    #     if self.trade_fill is not None:
    #     # test @luffy
    #     # if self.position_executor is not None:
    #         return self.get_strategy_summary()
    #     else:
    #         return None

    # def get_strategy_summary(self):
    
    def get_single_market_strategy_data(self, exchange: str, trading_pair: str):
        if self.trade_fill is None:
            return None
        
        if self.orders is not None:
            orders = self.orders[(self.orders["market"] == exchange) & (self.orders["symbol"] == trading_pair)].copy()
            trade_fill = self.trade_fill[self.trade_fill["order_id"].isin(orders["id"])].copy()
        else:
            orders = None
            trade_fill = self.trade_fill[(self.trade_fill["market"] == exchange) & (self.trade_fill["symbol"] == trading_pair)].copy()  

        if self.order_status is not None:
            order_status = self.order_status[self.order_status["order_id"].isin(orders["id"])].copy()
        else:
            order_status = None

        if self.market_data is not None:
            market_data = self.market_data[(self.market_data["exchange"] == exchange) &
                                           (self.market_data["trading_pair"] == trading_pair)].copy()
        else:
            market_data = None

        if self.position_executor is not None:
            position_executor = self.position_executor[(self.position_executor["exchange"] == exchange) &
                                                       (self.position_executor["trading_pair"] == trading_pair)].copy()
        else:
            position_executor = None

        return LoopySingleMarketStrategyData(
            exchange=exchange,
            trading_pair=trading_pair,
            orders=orders,
            order_status=order_status,
            trade_fill=trade_fill,
            market_data=market_data,
            position_executor=position_executor
        )
    
    @property
    def start_time(self):
        if self.orders is not None:
            return self.orders["creation_timestamp"].min()
        else:
            return self.trade_fill["timestamp"].min()

    @property
    def end_time(self):
        if self.orders is not None:
            return self.orders["last_update_timestamp"].max()
        else:
            return self.trade_fill["timestamp"].max()
    
class LoopySingleMarketStrategyData(SingleMarketStrategyData):

    def get_filtered_strategy_data(self, start_date: datetime.datetime, end_date: datetime.datetime):
        if self.orders is not None:
            orders = self.orders[
                (self.orders["creation_timestamp"] >= start_date) & (self.orders["creation_timestamp"] <= end_date)].copy()
            trade_fill = self.trade_fill[self.trade_fill["order_id"].isin(orders["id"])].copy()
        else:
            trade_fill = self.trade_fill[
                (self.trade_fill["timestamp"] >= start_date) & (self.trade_fill["timestamp"] <= end_date)].copy()
        
        if self.order_status is not None:
            order_status = self.order_status[self.order_status["order_id"].isin(orders["id"])].copy()
        else:
            order_status = None

        if self.market_data is not None:
            market_data = self.market_data[
                (self.market_data.index >= start_date) & (self.market_data.index <= end_date)].copy()
        else:
            market_data = None
        if self.position_executor is not None:
            position_executor = self.position_executor[(self.position_executor.datetime >= start_date) &
                                                       (self.position_executor.datetime <= end_date)].copy()
        else:
            position_executor = None

        return LoopySingleMarketStrategyData(
            exchange=self.exchange,
            trading_pair=self.trading_pair,
            orders=orders,
            order_status=order_status,
            trade_fill=trade_fill,
            market_data=market_data,
            position_executor=position_executor
        )
    
    # def get_market_data_resampled(self, interval):
    #     data_resampled = self.market_data.resample(interval).agg({
    #         "mid_price": "ohlc",
    #         "best_bid": "last",
    #         "best_ask": "last",
    #     })
    #     data_resampled.columns = data_resampled.columns.droplevel(0)
    #     return data_resampled
    
    @property
    def start_time(self):
        if self.orders is not None:
            return self.orders["creation_timestamp"].min()
        else:
            return self.trade_fill["timestamp"].min()

    @property
    def end_time(self):
        if self.orders is not None:
            return self.orders["last_update_timestamp"].max()
        else:
            return self.trade_fill["timestamp"].max()
        
    @property
    def total_buy_amount(self):
        return self.buys["base_amount"].sum()

    @property
    def total_sell_amount(self):
        return self.sells["base_amount"].sum()

    @property
    def average_buy_price(self):
        if self.total_buy_amount != 0:
            # average_price = (self.buys["price"] * self.buys["amount"]).sum() / self.total_buy_amount
            average_price = self.buys["quote_volume"].sum() / self.total_buy_amount
            return np.nan_to_num(average_price, nan=0)
        else:
            return 0

    @property
    def average_sell_price(self):
        if self.total_sell_amount != 0:
            # average_price = (self.sells["price"] * self.sells["amount"]).sum() / self.total_sell_amount
            average_price = self.sells["quote_volume"].sum() / self.total_sell_amount
            return np.nan_to_num(average_price, nan=0)
        else:
            return 0

    @property
    def trade_pnl_quote(self):
        # buy_volume = self.buys["amount"].sum() * self.average_buy_price
        # sell_volume = self.sells["amount"].sum() * self.average_sell_price
        buy_volume = self.buys["quote_volume"].sum()
        sell_volume = self.sells["quote_volume"].sum()
        inventory_change_volume = self.inventory_change_base_asset * self.end_price
        return sell_volume - buy_volume + inventory_change_volume

    @property
    def cum_fees_in_quote(self):
        return self.trade_fill["trade_fee_in_quote"].sum()

    @property
    def net_pnl_quote(self):
        return self.trade_pnl_quote - self.cum_fees_in_quote

    # @property
    # def inventory_change_base_asset(self):
    #     return self.total_buy_amount - self.total_sell_amount

    # @property
    # def accuracy(self):
    #     total_wins = (self.trade_fill["net_realized_pnl"] >= 0).sum()
    #     total_losses = (self.trade_fill["net_realized_pnl"] < 0).sum()
    #     return total_wins / (total_wins + total_losses)

    # @property
    # def profit_factor(self):
    #     total_profit = self.trade_fill.loc[self.trade_fill["realized_pnl"] >= 0, "realized_pnl"].sum()
    #     total_loss = self.trade_fill.loc[self.trade_fill["realized_pnl"] < 0, "realized_pnl"].sum()
    #     return total_profit / -total_loss
    

    

    