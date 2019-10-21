import psycopg2
from peewee import *

db = PostgresqlDatabase(database='exteneded_renko', user='postgres',
                        password='1', host='localhost')


class TradeSession(Model):
    algo_name = CharField()
    order_volume = FloatField(null=True)
    stop_loss = FloatField(null=True)
    take_profit = FloatField(null=True)
    net_profit = FloatField(null=True)
    trades_closed = FloatField(null=True)
    percentage_profit = FloatField(null=True)
    profit_factor = FloatField(null=True)
    max_drawdown = FloatField(null=True)
    avarage_trade = FloatField(null=True)
    date = DateTimeField(null=True)

    class Meta:
        database = db


class Position(Model):
    trade_session = ForeignKeyField(TradeSession)
    type = CharField(choices=(('Backtest', 'Backtest'), ('Live', 'Live')))
    started = DateTimeField()
    open_price = FloatField()
    finished = DateTimeField(null=True)
    closed_price = FloatField(null=True)
    amount = FloatField()
    status = CharField(choices=(('Closed', 'Closed'), ('Open', 'Open')))
    closed_by = CharField(null=True, choices=(('Algo', 'Algo'), ('Stop Loss', 'Stop Loss'), ('Take Profit', 'Take Profit'), (None, None)))
    pair = CharField()
    side = CharField(choices=(('Sell', 'Sell'), ('Buy', 'Buy')))
    exchange = CharField()
    timestamp = TimestampField()

    class Meta:
        database = db


class OHLCV(Model):
    exchange = CharField()
    pair = CharField()
    open = FloatField()
    high = FloatField()
    low = FloatField()
    close = FloatField()
    volume = FloatField()
    timestamp = TimestampField()



    class Meta:
        database = db


def connection(database):
    database.connect()
    database.create_tables([TradeSession, Position, OHLCV])


def trade_session(database, date="2018-01-01 00:00:00", order_volume=None, stop_loss=None, take_profit=None, net_profit=None, trades_closed=None,
                  percentage_profit=None, profit_factor=None, max_drawdown=None, avarage_trade=None):

    with database.atomic():
        global algo
        algo = TradeSession.create(algo_name='RenkoBB',
                                   order_volume=order_volume,
                                   stop_loss=stop_loss,
                                   take_profit=take_profit,
                                   net_profit=net_profit,
                                   trades_closed=trades_closed,
                                   percentage_profit=percentage_profit,
                                   profit_factor=profit_factor,
                                   max_drawdown=max_drawdown,
                                   avarage_trade=avarage_trade,
                                   date=date)


def positions(database, type, start, open_price, status, side, finish, close, amount, closed_by, exchange, timestamp):
    with database.atomic():
        pos = Position.create(
            trade_session=algo,
            type=type,
            started=start,
            open_price=open_price,
            finished=finish,
            closed_price=close,
            amount=amount,
            status=status,
            closed_by=closed_by,
            pair='BTCUSDT',
            side=side,
            exchange=exchange,
            timestamp=timestamp)


def get_ohlcv(database, exchange, pair, timestamp, open, high, low, close, volume):
    with database.atomic():
        ohlcv = OHLCV.create(
            exchange=exchange,
            pair=pair,
            open=open,
            high=high,
            low=low,
            close=close,
            volume=volume,
            timestamp=timestamp)

