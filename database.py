import psycopg2
from peewee import *

db = PostgresqlDatabase(database='exteneded_renko', user='postgres',
                        password='1', host='localhost')


class TradeSession(Model):
    algo_name = CharField()
    order_volume = FloatField(null=True)
    stop_loss = FloatField(null=True)
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
    name = ForeignKeyField(TradeSession)
    is_live = BooleanField()
    started = DateTimeField()
    open_price = FloatField()
    finished = DateTimeField()
    closed_price = FloatField()
    amount = FloatField()
    closed_by = CharField()
    pair = CharField()
    side = CharField()
    exchange = CharField()
    timestamp = TimestampField()

    class Meta:
        database = db


class OHLCV(Model):
    open = FloatField()
    high = FloatField()
    low = FloatField()
    close = FloatField()
    volume = FloatField()


    class Meta:
        database = db


def connection(database):
    database.connect()
    database.create_tables([TradeSession, Position, OHLCV])


def trade_session(database, date="2018-01-01 02:50:00", order_volume=None, stop_loss=None, net_profit=None, trades_closed=None,
                  percentage_profit=None, profit_factor=None, max_drawdown=None, avarage_trade=None):

    with database.atomic():
        global algo
        algo = TradeSession.create(algo_name='RenkoBB',
                                   order_volume=order_volume,
                                   stop_loss=stop_loss,
                                   net_profit=net_profit,
                                   trades_closed=trades_closed,
                                   percentage_profit=percentage_profit,
                                   profit_factor=profit_factor,
                                   max_drawdown=max_drawdown,
                                   avarage_trade=avarage_trade,
                                   date=date)


def positions(database, live, start, open_price, finish, close, amount, closed_by, exchange, timestamp):
    with database.atomic():
        pos = Position.create(
            name=algo,
            is_live=live,
            started=start,
            open_price=open_price,
            finished=finish,
            closed_price=close,
            amount=amount,
            closed_by=closed_by,
            pair='BTCUSDT',
            side='Buy',
            exchange=exchange,
            timestamp=timestamp)


def get_ohlcv(database, open, high, low, close, volume):
    with database.atomic():
        ohlcv = OHLCV.create(
            open=open,
            high=high,
            low=low,
            close=close,
            volume=volume)

