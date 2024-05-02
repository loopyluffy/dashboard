import pandas as pd
import pandas_ta as ta  # noqa: F401
from typing import Union, List
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from data_viz.dtypes import IndicatorConfig
from data_viz.tracers import PandasTAPlotlyTracer, PerformancePlotlyTracer
# from data_viz.candles import CandlesBase
from data_viz.performance.performance_candles import PerformanceCandles
# from utils.data_manipulation import StrategyData, SingleMarketStrategyData

from loopy_quant.loopy_candles import LoopyTAPlotlyTracer, LoopyCandles
from loopy_quant.loopy_data_manipulation import LoopyStrategyData, LoopySingleMarketStrategyData


class LoopyPerformanceCandles(LoopyCandles):
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
        # self.indicators_tracer = PandasTAPlotlyTracer(candles_df)
        self.indicators_tracer = LoopyTAPlotlyTracer(candles_df)
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
        configs = [
            IndicatorConfig(visible=True, title="bbands", row=1, col=1, color="blue", length=20, std=2.0),
            # IndicatorConfig(visible=True, title="ema", row=1, col=1, color="yellow", length=20),
            # IndicatorConfig(visible=True, title="ema", row=1, col=1, color="yellow", length=40),
            # IndicatorConfig(visible=True, title="ema", row=1, col=1, color="yellow", length=60),
            # IndicatorConfig(visible=True, title="ema", row=1, col=1, color="yellow", length=80),
            IndicatorConfig(visible=True, title="macd", row=2, col=1, color="red", fast=12, slow=26, signal=9),
            # IndicatorConfig(visible=True, title="macd", row=2, col=1, color="red", fast=9, slow=26, signal=13),
            IndicatorConfig(visible=True, title="macd_mc", row=2, col=1, color="blue", length=9),
            IndicatorConfig(visible=True, title="atr", row=4, col=1, color="red", length=9),
            # IndicatorConfig(visible=True, title="rsi", row=3, col=1, color="green", length=14)
        ]

        self.indicators_config = configs

    @property
    def buys(self):
        df = self.positions[["datetime", "entry_price", "close_price", "close_datetime", "side"]].copy()
        if len(df) > 0:
            df["price"] = df.apply(lambda row: row["entry_price"] if row["side"] == 1 else row["close_price"], axis=1)
            df["timestamp"] = df.apply(lambda row: row["datetime"] if row["side"] == 1 else row["close_datetime"], axis=1)
            df.set_index("timestamp", inplace=True)
            return df["price"]

    @property
    def sells(self):
        df = self.positions[["datetime", "entry_price", "close_price", "close_datetime", "side"]].copy()
        if len(df) > 0:
            df["price"] = df.apply(lambda row: row["entry_price"] if row["side"] == -1 else row["close_price"], axis=1)
            df["timestamp"] = df.apply(lambda row: row["datetime"] if row["side"] == -1 else row["close_datetime"], axis=1)
            df.set_index("timestamp", inplace=True)
            return df["price"]

    def get_n_rows_and_heights(self):
        rows = 1
        if self.show_indicators and self.indicators_config is not None:
            rows = max([config.row for config in self.indicators_config])
        if self.show_pnl:
            rows += 1
        if self.show_quote_inventory_change:
            rows += 1
        complementary_height = 1 - self.main_height
        row_heights = [self.main_height] + [round(complementary_height / (rows - 1), 3)] * (rows - 1) if rows > 1 else [1]
        return rows, row_heights

    def add_positions(self):
        if self.executor_version == "v1":
            for index, rown in self.positions.iterrows():
                if abs(rown["net_pnl_quote"]) > 0:
                    self.base_figure.add_trace(self.tracer.get_positions_traces(position_number=rown["id"],
                                                                                open_time=rown["datetime"],
                                                                                close_time=rown["close_datetime"],
                                                                                open_price=rown["entry_price"],
                                                                                close_price=rown["close_price"],
                                                                                side=rown["side"],
                                                                                close_type=rown["close_type"],
                                                                                stop_loss=rown["sl"], take_profit=rown["tp"],
                                                                                time_limit=rown["tl"],
                                                                                net_pnl_quote=rown["net_pnl_quote"]),
                                               row=1, col=1)
        elif self.executor_version == "v2":
            for index, rown in self.positions.iterrows():
                if abs(rown["net_pnl_quote"]) > 0:
                    self.base_figure.add_trace(self.tracer.get_positions_traces(position_number=rown["id"],
                                                                                open_time=rown["datetime"],
                                                                                close_time=rown["close_datetime"],
                                                                                open_price=rown["bep"],
                                                                                close_price=rown["close_price"],
                                                                                side=rown["side"],
                                                                                close_type=rown["close_type"],
                                                                                stop_loss=rown["sl"], take_profit=rown["tp"],
                                                                                time_limit=rown["tl"],
                                                                                net_pnl_quote=rown["net_pnl_quote"]),
                                               row=1, col=1)

    def add_dca_prices(self):
        if self.executor_version == "v2":
            for index, rown in self.positions.iterrows():
                if abs(rown["net_pnl_quote"]) > 0:
                    data = list(json.loads(rown["config"])["prices"])
                    data_series = pd.Series(data, index=[rown["datetime"]] * len(data))
                    self.base_figure.add_trace(self.tracer.get_entry_traces(data=data_series))
                    self.base_figure.add_vline(x=rown["datetime"], row=1, col=1, line_color="gray", line_dash="dash")


    def update_layout(self):
        self.base_figure.update_layout(
            legend=dict(
                orientation="h",
                x=0.5,
                y=1.04,
                xanchor="center",
                yanchor="bottom"
            ),
            height=1000,
            xaxis=dict(rangeslider_visible=False,
                       range=[self.min_time, self.max_time]),
            yaxis=dict(range=[self.candles_df.low.min(), self.candles_df.high.max()]),
            hovermode='x unified'
        )
        self.base_figure.update_yaxes(title_text="Price", row=1, col=1)
        self.base_figure.update_xaxes(title_text="Time", row=self.rows, col=1)
    