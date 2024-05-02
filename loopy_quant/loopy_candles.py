import pandas as pd
import pandas_ta as ta  # noqa: F401
from typing import Union, List
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from data_viz.dtypes import IndicatorConfig
from data_viz.tracers import PandasTAPlotlyTracer, PerformancePlotlyTracer
from data_viz.candles import CandlesBase
# from data_viz.performance.performance_candles import PerformanceCandles
# from data_viz.backtesting.backtesting_candles import BacktestingCandles

from loopy_quant.loopy_data_manipulation import LoopyStrategyData, LoopySingleMarketStrategyData

# example_case = [
#     IndicatorConfig(visible=True, title="bbands", row=1, col=1, color="blue", length=20, std=2.0),
#     IndicatorConfig(visible=True, title="ema", row=1, col=1, color="yellow", length=20),
#     IndicatorConfig(visible=True, title="ema", row=1, col=1, color="yellow", length=40),
#     IndicatorConfig(visible=True, title="ema", row=1, col=1, color="yellow", length=60),
#     IndicatorConfig(visible=True, title="ema", row=1, col=1, color="yellow", length=80),
#     IndicatorConfig(visible=True, title="macd", row=2, col=1, color="red", fast=12, slow=26, signal=9),
#     IndicatorConfig(visible=True, title="rsi", row=3, col=1, color="green", length=14)
# ]


class LoopyTAPlotlyTracer(PandasTAPlotlyTracer):
    def get_macd_traces(self, indicator_config):
        fast = indicator_config.fast
        slow = indicator_config.slow
        signal = indicator_config.signal

        # make a color difference @luffy
        color = indicator_config.color
        if color == 'blue':
            rgb = (0, 0, 255)
        elif color == 'green':
            rgb = (0, 255, 0)
        elif color == 'red':
            rgb = (255, 0, 0)
        else:
            rgb = (0, 0, 0)

        lighter_color = self.adjust_rgb(rgb, 100)
        darker_color = self.adjust_rgb(rgb, -100)

        if len(self.candles_df) < any([fast, slow, signal]):
            self.raise_error_if_not_enough_data(indicator_config.title)
        else:
            self.candles_df.ta.macd(fast=fast, slow=slow, signal=signal, append=True)
            macd_trace = go.Scatter(x=self.candles_df.index,
                                    y=self.candles_df[f'MACD_{fast}_{slow}_{signal}'],
                                    name=f'MACD_{fast}_{slow}_{signal}',
                                    mode='lines',
                                    line=dict(color=f'rgb{darker_color}', width=1))
            macd_signal_trace = go.Scatter(x=self.candles_df.index,
                                           y=self.candles_df[f'MACDs_{fast}_{slow}_{signal}'],
                                           name=f'MACDs_{fast}_{slow}_{signal}',
                                           mode='lines',
                                           line=dict(color=f'rgb{lighter_color}', width=1))
            macd_hist_trace = go.Bar(x=self.candles_df.index,
                                     y=self.candles_df[f'MACDh_{fast}_{slow}_{signal}'],
                                     name=f'MACDh_{fast}_{slow}_{signal}',
                                     marker=dict(color=indicator_config.color))
            return macd_trace, macd_signal_trace, macd_hist_trace
        
    def get_macd_mc_traces(self, indicator_config: IndicatorConfig):
        # macd_trace, macd_signal_trace, macd_hist_trace = super().get_macd_traces(indicator_config)
        # get a custom macd tracer
        length = indicator_config.length
        if len(self.candles_df) < length: #any([config.fast,]):
            # self.raise_error_if_not_enough_data(config.title)
            self.raise_error_if_not_enough_data("macd_mc")
            return
        else:
            self.candles_df[f"MACD_MC_{length}"] = (self.candles_df['open'] + self.candles_df['close']) / 2 - self.candles_df["close"].rolling(length).mean()
            self.candles_df[f"MACD_MC_RATIO_{length}"] = self.candles_df[f"MACD_MC_{length}"] / ((self.candles_df['open'] + self.candles_df['close']) / 2) * 100
            # self.candles_df[f"MACD_MC_RATIO_{length}"] = self.candles_df[f"MACD_MC_RATIO_{length}"] * 100
            macd_mc_trace = go.Scatter(x=self.candles_df.index,
                                    y=self.candles_df[f'MACD_MC_{length}'],
                                    name=f'MACD_MC_{length}',
                                    mode='lines',
                                    line=dict(color=indicator_config.color, width=1))
            macd_mc_ratio_trace = go.Scatter(x=self.candles_df.index,
                                    y=self.candles_df[f'MACD_MC_RATIO_{length}'],
                                    name=f'MACD_MC_RATIO_{length}',
                                    mode='lines',
                                    line=dict(color=indicator_config.color, width=1))
            # return macd_trace, macd_signal_trace, macd_hist_trace, macd_mc_trace
            return macd_mc_trace, macd_mc_ratio_trace
        
    # def get_atr_traces(self, length=9):
    def get_atr_traces(self, indicator_config: IndicatorConfig):
        length = indicator_config.length

         # make a color difference @luffy
        color = indicator_config.color
        if color == 'blue':
            rgb = (0, 0, 255)
        elif color == 'green':
            rgb = (0, 255, 0)
        elif color == 'red':
            rgb = (255, 0, 0)
        else:
            rgb = (0, 0, 0)

        lighter_color = self.adjust_rgb(rgb, 100)
        darker_color = self.adjust_rgb(rgb, -100)
        # print('-------------- color adjust -----------------------')
        # print(f'dark color: rgb({darker_color})')

        if len(self.candles_df) < length: #any([config.fast,]):
            # self.raise_error_if_not_enough_data(config.title)
            self.raise_error_if_not_enough_data("atr")
            return
        else:
            self.candles_df['TR'] = ta.true_range(self.candles_df['high'], self.candles_df['low'], self.candles_df['close'])
            self.candles_df[f"ATR_{length}"] = ta.atr(self.candles_df["high"], self.candles_df["low"], self.candles_df["close"], length=length)
            tr_trace = go.Scatter(x=self.candles_df.index,
                                    y=self.candles_df['TR'],
                                    name='TR',
                                    mode='lines',
                                    # line=dict(color=indicator_config.color, width=1))
                                    line=dict(color=f'rgb{darker_color}', width=1))
            atr_trace = go.Scatter(x=self.candles_df.index,
                                    y=self.candles_df[f'ATR_{length}'],
                                    name=f'ATR_{length}',
                                    mode='lines',
                                    # line=dict(color=indicator_config.color, width=1))
                                    line=dict(color=f'rgb{lighter_color}', width=1))
            return tr_trace, atr_trace
        
    def adjust_rgb(self, color, adjustment):
        """Adjust an RGB color to be lighter or darker.
        
        Parameters:
        - color: A tuple of (R, G, B) values, each in the range [0, 255].
        - adjustment: The amount to adjust each color component by. Positive
        values make the color lighter, while negative values make it darker.
        
        Returns:
        - A tuple of adjusted (R, G, B) values.

        Example usage:
        original_color = (100, 150, 200)  # A cool blue color
        lighter_color = adjust_rgb(original_color, adjustment=30)
        darker_color = adjust_rgb(original_color, adjustment=-30)
        """
        # Ensure the adjusted color values stay within the 0 to 255 range
        adjusted = [max(min(component + adjustment, 255), 0) for component in color]
        return tuple(adjusted)
    

class LoopyCandles(CandlesBase):
    # def __init__(self,
    #              candles_df: pd.DataFrame,
    #              indicators_config: List[IndicatorConfig] = None,
    #              show_annotations=True,
    #              line_mode=False,
    #              show_indicators=False,
    #              main_height=0.7,
    #              max_height=1000,
    #              rows: int = None,
    #              row_heights: list = None):
    def __init__(self,
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
                #  executor_version: str = "v1",
                 main_height: float = 0.7):
        self.candles_df = candles_df
        self.show_indicators = show_indicators
        self.indicators_config = indicators_config
        # self update indicator config... @luffy
        if show_indicators and indicators_config is None:
            self.add_indicator_config()
        self.show_annotations = show_annotations
        # self.indicators_tracer = PandasTAPlotlyTracer(candles_df)
        self.indicators_tracer = LoopyTAPlotlyTracer(candles_df)
        self.tracer = PerformancePlotlyTracer()
        self.line_mode = line_mode
        self.main_height = main_height
        self.max_height = max_height
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
        self.update_layout()

    def add_indicator_config(self):
        configs = [
            IndicatorConfig(visible=True, title="bbands", row=1, col=1, color="blue", length=20, std=2.0),
            # IndicatorConfig(visible=True, title="ema", row=1, col=1, color="yellow", length=20),
            # IndicatorConfig(visible=True, title="ema", row=1, col=1, color="yellow", length=40),
            # IndicatorConfig(visible=True, title="ema", row=1, col=1, color="yellow", length=60),
            # IndicatorConfig(visible=True, title="ema", row=1, col=1, color="yellow", length=80),
            IndicatorConfig(visible=True, title="macd", row=2, col=1, color="red", fast=12, slow=26, signal=9),
            # IndicatorConfig(visible=True, title="macd", row=2, col=1, color="red", fast=9, slow=26, signal=13),
            # IndicatorConfig(visible=True, title="macd_mc", row=2, col=1, color="blue", length=9),
            IndicatorConfig(visible=True, title="atr", row=4, col=1, color="red", length=9),
            # IndicatorConfig(visible=True, title="rsi", row=3, col=1, color="green", length=14)
        ]

        self.indicators_config = configs

    def add_macd_mc(self, indicator_config: IndicatorConfig):
        if indicator_config.visible:
            macd_mc_trace, macd_mc_ratio_trace = self.indicators_tracer.get_macd_mc_traces(indicator_config)
            self.base_figure.add_trace(trace=macd_mc_trace,
                                       row=indicator_config.row,
                                       col=indicator_config.col)
            self.base_figure.add_trace(trace=macd_mc_ratio_trace,
                                       row=indicator_config.row+1,
                                       col=indicator_config.col)
    
    def add_atr(self, indicator_config: IndicatorConfig):
        if indicator_config.visible:
            tr_trace, atr_trace = self.indicators_tracer.get_atr_traces(indicator_config)
            self.base_figure.add_trace(trace=tr_trace,
                                       row=indicator_config.row,
                                       col=indicator_config.col)
            self.base_figure.add_trace(trace=atr_trace,
                                       row=indicator_config.row,
                                       col=indicator_config.col)
    
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
            elif indicator.title == "macd_mc":
                self.add_macd_mc(indicator)
            elif indicator.title == "atr":
                self.add_atr(indicator)
            else:
                raise ValueError(f"{indicator.title} is not a valid indicator. Choose from bbands, ema, macd, rsi, macd_mc, atr")
        

    