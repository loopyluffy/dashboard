import pandas as pd
import pandas_ta as ta  # noqa: F401
from typing import Union, List
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from data_viz.dtypes import IndicatorsConfigBase, IndicatorConfig
from data_viz.tracers import PandasTAPlotlyTracer, PerformancePlotlyTracer
# from data_viz.candles import CandlesBase
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
    # def __init__(self,
    #              source: Union[LoopyStrategyData, LoopySingleMarketStrategyData],
    #              candles_df: pd.DataFrame = None,
    #              line_mode: bool = False,
    #              show_volume: bool = False,
    #              extra_rows: int = 2):
    #     # add indicator... @luffy
    #     self.indicators_config =  IndicatorsConfigBase()
    #     self.indicators_config.macd = IndicatorConfig(title='Indicator',row=2,col=1, fast=9, slow=26, signal=13)
    #     self.indicators_tracer = LoopyTAPlotlyTracer(candles_df, self.indicators_config)

    #     self.candles_df = candles_df
        
    #     self.tracer = PerformancePlotlyTracer()
    #     self.show_volume = show_volume
    #     self.line_mode = line_mode
    #     rows, heights = self.get_n_rows_and_heights(extra_rows)
    #     self.rows = rows
    #     specs = [[{"secondary_y": True}]] * rows
    #     self.base_figure = make_subplots(rows=rows,
    #                                      cols=1,
    #                                      shared_xaxes=True,
    #                                      vertical_spacing=0.005,
    #                                      row_heights=heights,
    #                                      specs=specs)
    #     if 'timestamp' in candles_df.columns:
    #         candles_df.set_index("timestamp", inplace=True)
    #     self.min_time = candles_df.index.min()
    #     self.max_time = candles_df.index.max()
    #     self.add_candles_graph()
    #     if self.show_volume:
    #         self.add_volume()
    #     if self.indicators_config is not None:
    #         self.add_indicators()

    #     # super().__init__(candles_df=self.candles_df,
    #     #                  indicators_config=None,
    #     #                  line_mode=line_mode,
    #     #                  show_volume=show_volume,
    #     #                  extra_rows=extra_rows)
    #     # CandlesBase.__init__(self, 
    #     #                      candles_df=self.candles_df,
    #     #                      indicators_config=indicator_config,
    #     #                      line_mode=line_mode,
    #     #                      show_volume=show_volume,
    #     #                      extra_rows=extra_rows)
            
    #     self.positions = source.position_executor
    #     self.add_buy_trades(data=self.buys)
    #     self.add_sell_trades(data=self.sells)
    #     self.add_positions()
    #     # self.add_pnl(data=source.trade_fill, realized_pnl_column="realized_trade_pnl", fees_column="cum_fees_in_quote",
    #     #              net_realized_pnl_column="net_realized_pnl", row_number=2)
    #     # self.add_quote_inventory_change(data=source.trade_fill, quote_inventory_change_column="inventory_cost",
    #     #                                 row_number=3)
    #     self.add_pnl(data=source.trade_fill, realized_pnl_column="realized_trade_pnl", fees_column="cum_fees_in_quote",
    #                  net_realized_pnl_column="net_realized_pnl", row_number=3)
    #     self.add_quote_inventory_change(data=source.trade_fill, quote_inventory_change_column="inventory_cost",
    #                                     row_number=4)
    #     self.update_layout()

    def __init__(self,
                #  source: Union[StrategyData, SingleMarketStrategyData],
                 source: Union[LoopyStrategyData, LoopySingleMarketStrategyData],
                 indicators_config: List[IndicatorConfig] = None,
                 candles_df: pd.DataFrame = None,
                 line_mode: bool = False,
                 show_buys: bool = False,
                 show_sells: bool = False,
                 show_positions: bool = False,
                 show_dca_prices: bool = False,
                 show_pnl: bool = True,
                 show_indicators: bool = False,
                 show_quote_inventory_change: bool = True,
                 show_annotations: bool = False,
                 executor_version: str = "v1",
                 main_height: float = 0.7):
        self.candles_df = candles_df

        self.positions = source.executors if executor_version == "v2" else source.position_executor
        self.executor_version = executor_version
        self.show_buys = show_buys
        self.show_sells = show_sells
        self.show_positions = show_positions
        self.show_pnl = show_pnl
        self.show_quote_inventory_change = show_quote_inventory_change
        self.show_indicators = show_indicators
        # self update indicator config... @luffy
        if indicators_config is not None:
            self.indicators_config = indicators_config
        elif show_indicators:
            self.add_indicator_config()
        self.main_height = main_height

        rows, row_heights = self.get_n_rows_and_heights()
        # CandlesBase.__init__ -------------------------------------------------------------------------
        # super().__init__(candles_df=self.candles_df,
        #                  indicators_config=indicators_config,
        #                  line_mode=line_mode,
        #                  show_indicators=show_indicators,
        #                  rows=rows,
        #                  row_heights=row_heights,
        #                  main_height=main_height,
        #                  show_annotations=show_annotations)
        # self.candles_df = candles_df
        # self.show_indicators = show_indicators
        # self.indicators_config = indicators_config
        self.show_annotations = show_annotations
        self.indicators_tracer = PandasTAPlotlyTracer(candles_df)
        # self.indicators_tracer = LoopyTAPlotlyTracer(candles_df)
        self.tracer = PerformancePlotlyTracer()
        self.line_mode = line_mode
        # self.main_height = main_height
        self.max_height = 1000
        self.rows = rows
        if rows is None:
            rows, row_heights = self.get_n_rows_and_heights()
            self.rows = rows
        specs = [[{"secondary_y": True}]] * self.rows
        self.base_figure = make_subplots(rows=self.rows,
                                         cols=1,
                                         shared_xaxes=True,
                                         vertical_spacing=0.005,
                                         row_heights=row_heights,
                                         specs=specs)
        if 'timestamp' in candles_df.columns:
            candles_df.set_index("timestamp", inplace=True)
        self.min_time = candles_df.index.min()
        self.max_time = candles_df.index.max()
        self.add_candles_graph()
        if self.show_indicators and self.indicators_config is not None:
            self.add_indicators()
        # self.update_layout()
        # ----------------------------------------------------------------------------------------------

        if show_buys:
            self.add_buy_trades(data=self.buys)
        if show_sells:
            self.add_sell_trades(data=self.sells)
        if show_positions:
            self.add_positions()
        if show_pnl:
            self.add_pnl(data=source.trade_fill,
                         realized_pnl_column="realized_trade_pnl",
                         fees_column="cum_fees_in_quote",
                         net_realized_pnl_column="net_realized_pnl",
                         row_number=rows - 1 if show_quote_inventory_change else rows)
        if show_quote_inventory_change:
            self.add_quote_inventory_change(data=source.trade_fill,
                                            quote_inventory_change_column="inventory_cost",
                                            row_number=rows)
        if show_dca_prices:
            self.add_dca_prices()
        self.update_layout()

    def add_indicator_config(self):
        # self.indicators_config =  IndicatorsConfigBase()
        # self.indicators_config.macd = IndicatorConfig(title='Indicator',row=2,col=1, fast=9, slow=26, signal=13)
        # self.indicators_tracer = LoopyTAPlotlyTracer(candles_df, self.indicators_config)

        configs = [
            IndicatorConfig(visible=True, title="bbands", row=1, col=1, color="blue", length=20, std=2.0),
            IndicatorConfig(visible=True, title="ema", row=1, col=1, color="yellow", length=20),
            IndicatorConfig(visible=True, title="ema", row=1, col=1, color="yellow", length=40),
            IndicatorConfig(visible=True, title="ema", row=1, col=1, color="yellow", length=60),
            IndicatorConfig(visible=True, title="ema", row=1, col=1, color="yellow", length=80),
            IndicatorConfig(visible=True, title="macd", row=2, col=1, color="red", fast=12, slow=26, signal=9),
            IndicatorConfig(visible=True, title="rsi", row=3, col=1, color="green", length=14)
        ]

        self.indicators_config = configs

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
    
    # def add_indicators(self):
    #     # self.add_bollinger_bands()
    #     # self.add_ema()
    #     # self.add_macd()
    #     self.add_macd_mc()
    #     self.add_atr()
    #     # self.add_rsi()
    def add_indicators(self):
        for indicator in self.indicators_config:
            if indicator.title == "bbands":
                self.add_bollinger_bands(indicator)
            elif indicator.title == "ema":
                self.add_ema(indicator)
            elif indicator.title == "macd":
                self.add_macd(indicator)
            elif indicator.title == "rsi":
                self.add_rsi(indicator)
            else:
                raise ValueError(f"{indicator.title} is not a valid indicator. Choose from bbands, ema, macd, rsi")

    