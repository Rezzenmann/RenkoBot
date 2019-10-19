import pandas as pd
import numpy as np
import talib
import pyrenko
from datetime import datetime
from logging import Logger
from catalyst import run_algorithm
from catalyst.api import symbol, order_target_percent, get_datetime, record, get_open_orders, get_order
from catalyst.exchange.utils.stats_utils import extract_transactions
import matplotlib.pyplot as plt

from database import *

NAMESPACE = 'RenkoBB'
log = Logger(NAMESPACE)


def initialize(context):
    context.i = 0
    context.asset = symbol('btc_usdt')
    context.tf = '1H'
    context.atr_time = 51
    context.bb = 8
    context.model = pyrenko.renko()

    context.close_trigger = [1, -1, -1]
    context.open_trigger = [-1, 1, 1]
    context.is_open = False
    context.num_trades = 0
    context.set_commission(maker=0.001, taker=0.002)
    context.set_slippage(slippage=0.001)
    context.order_price = None
    context.order_result = []
    context.close_price = None
    context.started = None
    context.finished = None
    context.amount = None
    context.ohlcv = {}


def handle_data(context, data):
    current_time = get_datetime().time()
    if current_time.hour == 0 and current_time.minute == 0:
        print('Current date is ' + str(get_datetime().date()))

    context.i += 1
    if context.i < context.atr_time:
        return

    if not data.can_trade(context.asset):
        return

    starting_cash = context.portfolio.starting_cash
    current = data.current(context.asset, 'close')
    price = data.current(context.asset, 'price')
    last_price = data.history(context.asset,
                              'price',
                              bar_count=context.atr_time - 1,
                              frequency=context.tf
                              )

    if context.i % 60 == 0:
        ohlcv_data = data.history(context.asset,
                                  fields=['open', 'high', 'low', 'close', 'volume'],
                                  bar_count=1,
                                  frequency='H')

        get_ohlcv(database=db, open=ohlcv_data.open, high=ohlcv_data.high,
        low=ohlcv_data.low, close=ohlcv_data.close,
        volume=ohlcv_data.volume)

    bb_data = data.history(context.asset,
                           'close',
                           bar_count=context.bb,
                           frequency=context.tf)

    hlc_data = data.history(context.asset,
                            fields=['high', 'low', 'close'],
                            bar_count=context.atr_time,
                            frequency=context.tf)



    upperband, middleband, lowerband = talib.BBANDS(bb_data, timeperiod=context.bb - 1, nbdevup=2, nbdevdn=2, matype=0)
    upperband, middleband, lowerband = upperband[-1], middleband[-1], lowerband[-1]
    bb_range = upperband - lowerband

    record(price=price,
           starting_cash = starting_cash,
           cash=context.portfolio.cash,
           upperband=upperband,
           middleband=middleband,
           lowerband=lowerband,
           num_trades=context.num_trades,
           order_result=context.order_result)

    context.model = pyrenko.renko()
    optimal_brick = context.model.set_brick_size(HLC_history=hlc_data)
    context.model.build_history(prices=last_price)

    prev_dir = context.model.get_renko_directions()
    last_dir = prev_dir[-4:-1]

    if not context.is_open:
        if last_dir == context.open_trigger and bb_range > 500:
            order_target_percent(context.asset, order_vol, limit_price=current*1.001)
            context.order_price = get_open_orders(context.asset)[-1].limit
            context.is_open = True
            context.started = get_open_orders(context.asset)[-1].dt
            context.order_price = get_open_orders(context.asset)[-1].limit
            context.amount = get_open_orders(context.asset)[-1].amount

    else:
        if current <= context.order_price * stop and stop != 0:
            close_id = order_target_percent(context.asset, 0, limit_price=current)
            context.is_open = False
            context.num_trades += 1
            price_diff = current - context.order_price
            context.order_result.append(price_diff)

            close_time = get_order(close_id).dt
            close_price = get_order(close_id).limit
            closed_by = 'Stop Loss'
            record(
                num_trades=context.num_trades,
                order_result=context.order_result
            )

            positions(db, live=live, start=context.started, open_price=context.order_price,
                      finish=close_time, close=close_price, amount=context.amount,
                      closed_by=closed_by, exchange=exchange_name, timestamp=datetime.timestamp(get_datetime()))
        else:
            if last_dir == context.close_trigger:
                close_id = order_target_percent(context.asset, 0, limit_price=current)
                context.model = pyrenko.renko()
                context.is_open = False

                price_diff = current - context.order_price
                context.order_result.append(price_diff)
                context.num_trades += 1
                close_time = get_order(close_id).dt
                close_price = get_order(close_id).limit
                closed_by = 'Algo'

                record(
                    num_trades=context.num_trades,
                    order_result=context.order_result
                )

                positions(db, live=live, start=context.started, open_price=context.order_price,
                          finish=close_time, close=close_price, amount=context.amount,
                          closed_by=closed_by, exchange=exchange_name, timestamp=datetime.timestamp(get_datetime()))


def analyze(context, perf):

    # print('Total return: ' + str(perf.algorithm_period_return[-1]))
    # print('Sortino coef: ' + str(perf.sortino[-1]))
    # print('Max drawdown: ' + str(np.min(perf.max_drawdown)))
    # print('Alpha: ' + str(perf.alpha[-1]))
    # print('Beta: ' + str(perf.beta[-1]))
    # print('Starting cash:' + str(perf.starting_cash[0]))
    # print('Ending cash:' + str(perf.cash[-1]))
    # print('Number of trades:' + str(int(perf.num_trades[-1])))

    positive_trades = []
    negative_trades = []

    net_profit = sum(perf.order_result[-1])
    profit_percent = (perf.starting_cash[0] + net_profit) / perf.starting_cash[0]

    for trade in perf.order_result[-1]:
        if trade >= 0:
            positive_trades.append(trade)
        else:
            negative_trades.append(trade)

    profit_factor = sum(positive_trades) / sum(negative_trades)

    if len(perf.order_result[-1]) > 0:
        average_trade = sum(perf.order_result[-1])/len(perf.order_result[-1])

    if order_vol == 0.1 and stop == 0.9975:
        q = (TradeSession
             .update({'order_volume': order_vol, 'stop_loss':stop, 'net_profit':net_profit,
                      'trades_closed':context.num_trades, 'avarage_trade':average_trade,
                      'percentage_profit':profit_percent, 'profit_factor':profit_factor,
                      'max_drawdown':np.min(perf.max_drawdown), 'date':datetime.now()})
             .where(TradeSession.net_profit is None))
        q.execute()

    else:

        trade_session(db, order_volume=order_vol, stop_loss=stop, net_profit=net_profit,
                      trades_closed=context.num_trades,
                      avarage_trade=average_trade, percentage_profit=profit_percent, profit_factor=profit_factor,
                      max_drawdown=np.min(perf.max_drawdown), date=datetime.now())

    # exchange = list(context.exchanges.values())[0]
    # quote_currency = exchange.quote_currency.upper()
    #
    # ax1 = plt.subplot(311)
    # perf.loc[:, ['price', 'upperband', 'middleband', 'lowerband']].plot(
    #     ax=ax1,
    #     label='Price')
    #
    # ax1.set_ylabel('{asset}\n({quote})'.format(
    #     asset=context.asset.symbol,
    #     quote=quote_currency
    # ))
    # start, end = ax1.get_ylim()
    # ax1.yaxis.set_ticks(np.arange(start, end, (end - start) / 5))
    #
    # ax2 = plt.subplot(312)
    # perf.loc[:, 'portfolio_value'].plot(ax=ax2)
    # ax2.set_ylabel('Portfolio\nValue\n({})'.format(quote_currency))
    #
    # ax3 = plt.subplot(313)
    # perf.loc[:, 'price'].plot(ax=ax3, label='Price')
    #
    # ax3.set_ylabel('{asset}\n({quote})'.format(
    #     asset=context.asset.symbol, quote=quote_currency
    # ))
    #
    # transaction_df = extract_transactions(perf)
    #
    # if not transaction_df.empty:
    #     buy_df = transaction_df[transaction_df['amount'] > 0]
    #     sell_df = transaction_df[transaction_df['amount'] < 0]
    #     ax3.scatter(
    #         buy_df.index.to_pydatetime(),
    #         perf.loc[buy_df.index.floor('1 min'), 'price'],
    #         marker='^',
    #         s=100,
    #         c='green',
    #         label=''
    #     )
    #
    #     ax3.scatter(
    #         sell_df.index.to_pydatetime(),
    #         perf.loc[sell_df.index.floor('1 min'), 'price'],
    #         marker='v',
    #         s=100,
    #         c='red',
    #           label=''
    #     )
    #
    # # uncomment to show charts after
    # # plt.show()


if __name__ == '__main__':

    order_volume = [0.1] # , 0.25, 0.5, 1]
    stop_loss = [0.9975, 0.995] # , 0.99 , 0.95, 0.925, 0.90, 0]

    connection(db)
    trade_session(db)

    start_date = '2018-1-1'
    end_date = '2018-1-2'

    exchange_name = 'binance'
    live = False


    for order_vol in order_volume:
        for stop in stop_loss:
            started=datetime.now()
            run_algorithm(
                capital_base=20000,
                data_frequency='minute',
                initialize=initialize,
                handle_data=handle_data,
                analyze=analyze,
                exchange_name=exchange_name,
                algo_namespace=NAMESPACE,
                quote_currency='usdt',
                live=False,
                start=pd.to_datetime(start_date, utc=True),
                end=pd.to_datetime(end_date, utc=True)
            )
