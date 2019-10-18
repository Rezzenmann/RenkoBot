import psycopg2
from peewee import *

db = PostgresqlDatabase(database='exteneded_renko', user='postgres',
                        password='1', host='localhost')


class TradeSession(Model):
    algo_name = CharField()
    order_volume = FloatField()
    stop_loss = FloatField()
    net_profit = FloatField()
    trades_closed = FloatField()
    percentage_profit = FloatField()
    profit_factor = FloatField()
    max_drawdown = FloatField()
    avarage_trade = FloatField()
    date = DateTimeField()

    class Meta:
        database = db


class Position(Model):
    name = CharField()
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

    class Meta:
        database = db


class OHLCV(Model):
    date = DateTimeField()
    open = FloatField()
    high = FloatField()
    low = FloatField()
    close = FloatField()
    volume = FloatField()

    class Meta:
        database = db


def connection(database):
    database.connect()
    database.create_tables([TradeSession, Position])


def positions(database, live, start, open_price, finish, close, amount, closed_by, exchange):
    with database.atomic():
        pos = Position.create(
            name='RenkoBB',
            is_live=live,
            started=start,
            open_price=open_price,
            finished=finish,
            closed_price=close,
            amount=amount,
            closed_by=closed_by,
            pair='BTCUSDT',
            side='Buy',
            exchange=exchange)


def trade_session(database, order_volume, stop_loss, net_profit, trades_closed,
                  percentage_profit, profit_factor, max_drawdown, avarage_trade,
                  date):

    with database.atomic():
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

