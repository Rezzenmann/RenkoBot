import pandas as pd
import numpy as np
import talib
import pyrenko
from logging import Logger
from catalyst import run_algorithm
from catalyst.api import symbol, order_target_percent, get_datetime, record
from catalyst.exchange.utils.stats_utils import extract_transactions
import matplotlib.pyplot as plt

import csv

NAMESPACE = 'RenkoBB'
log = Logger(NAMESPACE)


def write_cvs(data):
    if order_vol == order_volume[0]:
        with open('results.csv', 'a') as f:
            writer = csv.DictWriter(f, fieldnames=['OrderVolume', "TotalReturn", "StartingCash", "EndingCash",
                                                   'TradesClosed'])
            writer.writeheader()

    with open('results.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow((data['OrderVolume'], data['TotalReturn'], data['StartingCash'],
                         data['EndingCash'], data['TradesClosed']))


def initialize(context):
    context.i = 0
    context.asset = symbol('btc_usdt')
    context.tf = '1H'
    context.atr_time = 51
    context.bb = 7
    context.model = pyrenko.renko()

    context.close_trigger = [1, -1, -1]
    context.open_trigger = [-1, 1, 1]
    context.is_open = False
    context.num_trades = 0


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
           lowerband=lowerband)

    context.model = pyrenko.renko()
    optimal_brick = context.model.set_brick_size(HLC_history=hlc_data)
    context.model.build_history(prices=last_price)

    prev_dir = context.model.get_renko_directions()
    last_dir = prev_dir[-4:-1]

    if context.is_open == False:
        if last_dir == context.open_trigger and bb_range > 500:
        # if current <= lowerband:
            print('Block is ', optimal_brick)
            order_target_percent(context.asset, order_vol, limit_price=current*1.05)
            print('Position opened at {}'.format(current))
            context.is_open = True


    else:
        if last_dir == context.close_trigger:
        # if current >= upperband:
            print('Block is ', optimal_brick)
            print('Position closed at {}'.format(current))
            order_target_percent(context.asset, 0, limit_price=current*0.95)
            context.model = pyrenko.renko()
            context.is_open = False

            context.num_trades +=1

            record(
                num_trades=context.num_trades
            )


def analyze(context, perf):
    print('Total return: ' + str(perf.algorithm_period_return[-1]))
    print('Sortino coef: ' + str(perf.sortino[-1]))
    print('Max drawdown: ' + str(perf.max_drawdown))
    print('Alpha: ' + str(perf.alpha[-1]))
    print('Beta: ' + str(perf.beta[-1]))
    print('Starting cash:' + str(perf.starting_cash[0]))
    print('Ending cash:' + str(perf.cash[-1]))
    print('Number of trades:' + str(int(perf.num_trades[-1])))

    csv_data = {'OrderVolume': order_vol,
                'TotalReturn': perf.algorithm_period_return[-1],
                'StartingCash': perf.starting_cash[0],
                'EndingCash': perf.cash[-1],
                'TradesClosed': perf.num_trades[-1]}

    write_cvs(csv_data)

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
    #         label=''
    #     )
    # # plt.show()


if __name__ == '__main__':

    order_volume = [0.1, 0.25, 0.5, 1]

    for order_vol in order_volume:

        run_algorithm(
            capital_base=20000,
            data_frequency='minute',
            initialize=initialize,
            handle_data=handle_data,
            analyze=analyze,
            exchange_name='binance',
            algo_namespace=NAMESPACE,
            quote_currency='usdt',
            live=False,
            start=pd.to_datetime('2018-1-1', utc=True),
            end=pd.to_datetime('2018-1-7', utc=True)
        )