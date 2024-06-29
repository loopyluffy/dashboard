# import os
# import pandas as pd
import streamlit as st
# import math
# from datetime import date, timedelta
# change package @luffy
from loopy_quant.data.loopy_database_manager import get_databases
# from loopy_quant.loopy_real_db_manager import LoopyRealDBManager
from loopy_quant.data.loopy_backtesting_db_manager2 import LoopyBacktestingDBManager2
from loopy_quant.viz.loopy_candles import LoopyCandles

# from quants_lab.strategy.strategy_analysis import StrategyAnalysis
from data_viz.dtypes import IndicatorConfig
from data_viz.backtesting.backtesting_charts import BacktestingCharts
from data_viz.backtesting.backtesting_candles import BacktestingCandles
from utils.st_utils import initialize_st_page #, download_csv_button, style_metric_cards, db_error_message
# import data_viz.utils as utils


initialize_st_page(title="Loopy Backtesting Performance", icon="🔬", initial_sidebar_state="collapsed")
# initialize_st_page(title="Backtesting Performance", icon="🚀")
# style_metric_cards()

def initialize_session_state_vars():
    if "strategy_params" not in st.session_state:
        st.session_state.strategy_params = {}
    if "backtesting_params" not in st.session_state:
        st.session_state.backtesting_params = {}

# Data source section
st.subheader("🔫 Data source")

initialize_session_state_vars()
# Find and select existing databases
dbs = get_databases(backtesting=True)
selected_db = None
if dbs is not None:
    bot_source = st.selectbox("Choose your database source:", dbs.keys())
    db_names = [x for x in dbs[bot_source]]
    selected_db_name = st.selectbox("Select a database to start:", db_names)
    if "postgres" in dbs[bot_source][selected_db_name]:
        if "backtest" in dbs[bot_source][selected_db_name]:
            selected_db = LoopyBacktestingDBManager2(dbs[bot_source][selected_db_name])

if selected_db is None:
    st.warning("Ups! No databases were founded. Select a backtesting database")
    st.stop()

# Strategy summary section
if selected_db is not None:
    st.divider()
    st.subheader("📝 Strategy summary")

    # backtesting strategy list in a selected database
    strategy_list = selected_db.get_backtesting_strategy_list()
    if strategy_list is not None:
        selected_strategy = st.selectbox("Choose a backtesting strategy:", strategy_list)
        st.session_state["strategy_params"]["strategy"] = selected_strategy
    else:
        st.warning("Ups! No backtesting records!!")
        st.stop()

    # backtesting strategy summaries in a selected strategy
    strategy_summary = selected_db.get_backtesting_strategy_summary(selected_strategy)
    if strategy_summary is not None:
        strategy_summary["explore"] = False
        sorted_cols = [
                "explore",
                # "timestamp",
                "strategy",
                # "strategy_id",
                "exchange",
                "trading_pair",
                "interval",
                "start",
                "end",
                "initial_portfolio_usd",
                # "net_pnl",
                "net_pnl_usd",
                "trade_cost",
                "sl",
                "tp",
                "tl",
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
        strategy_summary = strategy_summary.reindex(columns=sorted_cols, fill_value=0)
        summary_table = st.data_editor(
                                strategy_summary,
                                column_config={"explore": st.column_config.CheckboxColumn(required=True)},
                                use_container_width=True,
                                hide_index=True
                        )
        # print("-------------summary table----------")
        # print(strategy_summary)
        selection = summary_table[summary_table.explore]
        # if selection is None:
        if len(selection) == 0:
            st.info("💡 Choose a strategy and start analyzing!")
            st.stop()
        elif len(selection) > 1:
            st.warning("This version doesn't support multiple selections. Please try selecting only one.")
            st.stop()
        else: # len(selection) == 1:
            selected_exchange = selection["exchange"].values[0]
            selected_trading_pair = selection["trading_pair"].values[0]
            selected_start = selection["start"].values[0]
            selected_end = selection["end"].values[0]
            st.session_state["strategy_params"]["exchange"] = selected_exchange
            st.session_state["strategy_params"]["trading_pair"] = selected_trading_pair

        strategy_analysis = selected_db.get_position_analysis(
                                            strategy=selected_strategy,
                                            exchange=selected_exchange,
                                            trading_pair=selected_trading_pair,
                                            start_date=selected_start, 
                                            end_date=selected_end
                            )
        if strategy_analysis is None:
            st.info("💡 Ups! No backtesting datum were founded!!!")
            st.stop()

# Visibility options
with st.expander("Visual Options"):
    # col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        show_buys = st.checkbox("Buys", value=True)
    with col2:
        show_sells = st.checkbox("Sells", value=True)
        # show_annotations = st.checkbox("Annotations", value=True)
    with col3:
        show_positions = st.checkbox("Positions", value=True)
    # with col4:
    #     show_pnl = st.checkbox("PNL", value=True)
    # with col5:
    #     show_quote_inventory_change = st.checkbox("Quote Inventory Change", value=False)
    with col4:
        show_indicators = st.checkbox("Indicators", value=True)
    # with col7:
    #     main_height = st.slider("Main Row Height", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
        
# configs = [
#     IndicatorConfig(visible=True, title="bbands", row=1, col=1, color="blue", length=20, std=2.0),
#     # IndicatorConfig(visible=True, title="ema", row=1, col=1, color="yellow", length=20),
#     # IndicatorConfig(visible=True, title="ema", row=1, col=1, color="yellow", length=40),
#     # IndicatorConfig(visible=True, title="ema", row=1, col=1, color="yellow", length=60),
#     # IndicatorConfig(visible=True, title="ema", row=1, col=1, color="yellow", length=80),
#     IndicatorConfig(visible=True, title="macd", row=2, col=1, color="red", fast=12, slow=26, signal=9),
#     # IndicatorConfig(visible=True, title="macd", row=2, col=1, color="red", fast=9, slow=26, signal=13),
#     # IndicatorConfig(visible=True, title="macd_mc", row=2, col=1, color="blue", length=9),
#     # IndicatorConfig(visible=True, title="atr", row=4, col=1, color="red", length=9),
#     # IndicatorConfig(visible=True, title="rsi", row=3, col=1, color="green", length=14)
# ]

backtesting_charts = BacktestingCharts(strategy_analysis)
# backtesting_candles = BacktestingCandles(strategy_analysis,
#                                         # indicators_config=utils.load_indicators_config(indicators_config_path),
#                                         # indicators_config=configs,
#                                         line_mode=False,
#                                         show_buys=show_buys,
#                                         show_sells=show_sells,
#                                         show_indicators=show_indicators,
#                                         show_positions=show_positions)
backtesting_candles = LoopyCandles(source=strategy_analysis,
                                   # indicators_config=utils.load_indicators_config(indicators_config_path),
                                   # indicators_config=configs,
                                   line_mode=False,
                                   show_buys=show_buys,
                                   show_sells=show_sells,
                                   show_indicators=show_indicators,
                                   show_positions=show_positions)

col1, col2 = st.columns(2)
with col1:
    st.subheader("🏦 General")
with col2:
    st.subheader("📋 General stats")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Exchange", st.session_state["strategy_params"]["exchange"])
with col2:
    st.metric("Trading Pair", st.session_state["strategy_params"]["trading_pair"])
with col3:
    st.metric("Start date", strategy_analysis.start_date().strftime("%Y-%m-%d %H:%M"))
    st.metric("End date", strategy_analysis.end_date().strftime("%Y-%m-%d %H:%M"))
with col4:
    st.metric("Duration (hours)", f"{strategy_analysis.duration_in_minutes() / 60:.2f}")
    st.metric("Price change", st.session_state["strategy_params"]["trading_pair"])
st.subheader("📈 Performance")
col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
with col1:
    st.metric("Net PnL USD",
                f"{strategy_analysis.net_profit_usd():.2f}",
                delta=f"{100 * strategy_analysis.net_profit_pct():.2f}%",
                help="The overall profit or loss achieved.")
with col2:
    st.metric("Total positions",
                f"{strategy_analysis.total_positions()}",
                help="The total number of closed trades, winning and losing.")
with col3:
    st.metric("Accuracy",
                f"{100 * (len(strategy_analysis.win_signals()) / strategy_analysis.total_positions()):.2f} %",
                help="The percentage of winning trades, the number of winning trades divided by the"
                    " total number of closed trades")
with col4:
    st.metric("Profit factor",
                f"{strategy_analysis.profit_factor():.2f}",
                help="The amount of money the strategy made for every unit of money it lost, "
                    "gross profits divided by gross losses.")
with col5:
    st.metric("Max Drawdown",
                f"{strategy_analysis.max_drawdown_usd():.2f}",
                delta=f"{100 * strategy_analysis.max_drawdown_pct():.2f}%",
                help="The greatest loss drawdown, i.e., the greatest possible loss the strategy had compared "
                    "to its highest profits")
with col6:
    st.metric("Avg Profit",
                f"{strategy_analysis.avg_profit():.2f}",
                help="The sum of money gained or lost by the average trade, Net Profit divided by "
                    "the overall number of closed trades.")
with col7:
    st.metric("Avg Minutes",
                f"{strategy_analysis.avg_trading_time_in_minutes():.2f}",
                help="The average number of minutes that elapsed during trades for all closed trades.")
with col8:
    st.metric("Sharpe Ratio",
                f"{strategy_analysis.sharpe_ratio():.2f}",
                help="The Sharpe ratio is a measure that quantifies the risk-adjusted return of an investment"
                    " or portfolio. It compares the excess return earned above a risk-free rate per unit of"
                    " risk taken.")
st.plotly_chart(backtesting_charts.realized_pnl_over_time_fig, use_container_width=True)
st.subheader("💱 Market activity")
st.plotly_chart(backtesting_candles.figure(), use_container_width=True)




