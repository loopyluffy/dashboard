import pandas as pd
import pandas_ta as ta  # noqa: F401
from typing import Union
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from data_viz.dtypes import IndicatorsConfigBase, IndicatorConfig
from data_viz.tracers import PandasTAPlotlyTracer, PerformancePlotlyTracer
from data_viz.candles_base import CandlesBase
from data_viz.performance.performance_candles import PerformanceCandles
# from utils.data_manipulation import StrategyData, SingleMarketStrategyData

from loopy_quant.loopy_data_manipulation import LoopyStrategyData, LoopySingleMarketStrategyData


class LoopyTAPlotlyTracer(PandasTAPlotlyTracer):
    def get_macd_mc_traces(self, length=9):
        config = self.indicators_config.macd.copy()
        length = config.fast
        if len(self.candles_df) < length: #any([config.fast,]):
            # self.raise_error_if_not_enough_data(config.title)
            self.raise_error_if_not_enough_data("macd_mc")
            return
        else:
            self.candles_df[f"MACD_MC_{length}"] = (self.candles_df['open'] + self.candles_df['close']) / 2 - self.candles_df["close"].rolling(length).mean()
            macd_mc_trace = go.Scatter(x=self.candles_df.index,
                                    y=self.candles_df[f'MACD_MC_{length}'],
                                    name=f'MACD_MC_{length}',
                                    mode='lines',
                                    line=dict(color=config.color, width=1))
            return macd_mc_trace
        
    def get_atr_traces(self, length=9):
        config = self.indicators_config.macd.copy()
        length = config.fast
        if len(self.candles_df) < length: #any([config.fast,]):
            # self.raise_error_if_not_enough_data(config.title)
            self.raise_error_if_not_enough_data("macd_mc")
            return
        else:
            self.candles_df['TR'] = ta.true_range(self.candles_df['high'], self.candles_df['low'], self.candles_df['close'])
            self.candles_df[f"ATR_{length}"] = ta.atr(self.candles_df["high"], self.candles_df["low"], self.candles_df["close"], length=length)
            tr_trace = go.Scatter(x=self.candles_df.index,
                                    y=self.candles_df['TR'],
                                    name='TR',
                                    mode='lines',
                                    # line=dict(color=config.color, width=1))
                                    line=dict(color='red', width=1))
            atr_trace = go.Scatter(x=self.candles_df.index,
                                    y=self.candles_df[f'ATR_{length}'],
                                    name=f'ATR_{length}',
                                    mode='lines',
                                    # line=dict(color=config.color, width=1))
                                    line=dict(color='blue', width=1))
            return tr_trace, atr_trace
        

class LoopyPerformanceCandles(PerformanceCandles):
# class LoopyPerformanceCandles(CandlesBase):
    def __init__(self,
                 source: Union[LoopyStrategyData, LoopySingleMarketStrategyData],
                 candles_df: pd.DataFrame = None,
                 line_mode: bool = False,
                 show_volume: bool = False,
                 extra_rows: int = 2):
        # add indicator... @luffy
        self.indicators_config =  IndicatorsConfigBase()
        self.indicators_config.macd = IndicatorConfig(title='Indicator',row=2,col=1, fast=9, slow=26, signal=13)
        self.indicators_tracer = LoopyTAPlotlyTracer(candles_df, self.indicators_config)

        self.candles_df = candles_df
        
        self.tracer = PerformancePlotlyTracer()
        self.show_volume = show_volume
        self.line_mode = line_mode
        rows, heights = self.get_n_rows_and_heights(extra_rows)
        self.rows = rows
        specs = [[{"secondary_y": True}]] * rows
        self.base_figure = make_subplots(rows=rows,
                                         cols=1,
                                         shared_xaxes=True,
                                         vertical_spacing=0.005,
                                         row_heights=heights,
                                         specs=specs)
        if 'timestamp' in candles_df.columns:
            candles_df.set_index("timestamp", inplace=True)
        self.min_time = candles_df.index.min()
        self.max_time = candles_df.index.max()
        self.add_candles_graph()
        if self.show_volume:
            self.add_volume()
        if self.indicators_config is not None:
            self.add_indicators()

        # super().__init__(candles_df=self.candles_df,
        #                  indicators_config=None,
        #                  line_mode=line_mode,
        #                  show_volume=show_volume,
        #                  extra_rows=extra_rows)
        # CandlesBase.__init__(self, 
        #                      candles_df=self.candles_df,
        #                      indicators_config=indicator_config,
        #                      line_mode=line_mode,
        #                      show_volume=show_volume,
        #                      extra_rows=extra_rows)
            
        self.positions = source.position_executor
        self.add_buy_trades(data=self.buys)
        self.add_sell_trades(data=self.sells)
        self.add_positions()
        # self.add_pnl(data=source.trade_fill, realized_pnl_column="realized_trade_pnl", fees_column="cum_fees_in_quote",
        #              net_realized_pnl_column="net_realized_pnl", row_number=2)
        # self.add_quote_inventory_change(data=source.trade_fill, quote_inventory_change_column="inventory_cost",
        #                                 row_number=3)
        self.add_pnl(data=source.trade_fill, realized_pnl_column="realized_trade_pnl", fees_column="cum_fees_in_quote",
                     net_realized_pnl_column="net_realized_pnl", row_number=3)
        self.add_quote_inventory_change(data=source.trade_fill, quote_inventory_change_column="inventory_cost",
                                        row_number=4)
        self.update_layout()

    def add_macd_mc(self):
        if self.indicators_config.macd.visible:
            macd_mc_trace = self.indicators_tracer.get_macd_mc_traces()
            self.base_figure.add_trace(trace=macd_mc_trace,
                                       row=self.indicators_config.macd.row,
                                       col=self.indicators_config.macd.col)
    
    def add_atr(self):
        if self.indicators_config.macd.visible:
            tr_trace, atr_trace = self.indicators_tracer.get_atr_traces()
            self.base_figure.add_trace(trace=tr_trace,
                                       row=self.indicators_config.macd.row,
                                       col=self.indicators_config.macd.col)
            self.base_figure.add_trace(trace=atr_trace,
                                       row=self.indicators_config.macd.row,
                                       col=self.indicators_config.macd.col)
    
    def add_indicators(self):
        # self.add_bollinger_bands()
        # self.add_ema()
        # self.add_macd()
        self.add_macd_mc()
        self.add_atr()
        # self.add_rsi()

    # @property
    # def buys(self):
    #     df = self.positions[["datetime", "entry_price", "close_price", "close_datetime", "side"]].copy()
    #     df["price"] = df.apply(lambda row: row["entry_price"] if row["side"] == 1 else row["close_price"], axis=1)
    #     df["timestamp"] = df.apply(lambda row: row["datetime"] if row["side"] == 1 else row["close_datetime"], axis=1)
    #     df.set_index("timestamp", inplace=True)
    #     return df["price"]

    # @property
    # def sells(self):
    #     df = self.positions[["datetime", "entry_price", "close_price", "close_datetime", "side"]].copy()
    #     df["price"] = df.apply(lambda row: row["entry_price"] if row["side"] == -1 else row["close_price"], axis=1)
    #     df["timestamp"] = df.apply(lambda row: row["datetime"] if row["side"] == -1 else row["close_datetime"], axis=1)
    #     df.set_index("timestamp", inplace=True)
    #     return df["price"]

    # def add_positions(self):
    #     for index, rown in self.positions.iterrows():
    #         self.base_figure.add_trace(self.tracer.get_positions_traces(position_number=rown["id"],
    #                                                                     open_time=rown["datetime"],
    #                                                                     close_time=rown["close_datetime"],
    #                                                                     open_price=rown["entry_price"],
    #                                                                     close_price=rown["close_price"],
    #                                                                     side=rown["side"],
    #                                                                     close_type=rown["close_type"],
    #                                                                     stop_loss=rown["sl"], take_profit=rown["tp"],
    #                                                                     time_limit=rown["tl"],
    #                                                                     net_pnl_quote=rown["net_pnl_quote"]),
    #                                    row=1, col=1)

    # def update_layout(self):
    #     self.base_figure.update_layout(
    #         title={
    #             'text': "Market activity",
    #             'y': 0.99,
    #             'x': 0.5,
    #             'xanchor': 'center',
    #             'yanchor': 'top'
    #         },
    #         legend=dict(
    #             orientation="h",
    #             x=0.5,
    #             y=1.04,
    #             xanchor="center",
    #             yanchor="bottom"
    #         ),
    #         height=1000,
    #         xaxis=dict(rangeslider_visible=False,
    #                    range=[self.min_time, self.max_time]),
    #         yaxis=dict(range=[self.candles_df.low.min(), self.candles_df.high.max()]),
    #         hovermode='x unified'
    #     )
    #     self.base_figure.update_yaxes(title_text="Price", row=1, col=1)
    #     if self.show_volume:
    #         self.base_figure.update_yaxes(title_text="Volume", row=2, col=1)
    #     self.base_figure.update_xaxes(title_text="Time", row=self.rows, col=1)
