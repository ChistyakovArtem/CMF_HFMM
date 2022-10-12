from dataclasses import dataclass
from typing import Optional
import pandas as pd
from collections import deque
import time


@dataclass
class Order:  # Our own placed order
    timestamp: float
    order_id: int
    side: str
    size: float
    price: float


@dataclass
class AnonTrade:  # Market trade
    timestamp: float
    side: str
    size: float
    price: str


@dataclass
class OwnTrade:  # Execution of own placed order
    timestamp: float
    order_id: int
    side: str
    size: float
    price: float


@dataclass
class OrderbookSnapshotUpdate:  # Orderbook tick snapshot
    timestamp: float
    ask_price: float
    bid_price: float


class DataLoader:
    def __init__(self):
        btc_obs = pd.read_csv('data/md/btcusdt_Binance_LinearPerpetual/lobs.csv')
        btc_trades = pd.read_csv('data/md/btcusdt_Binance_LinearPerpetual/trades.csv')

        ask_prefix = 'btcusdt:Binance:LinearPerpetual_ask_'
        bid_prefix = 'btcusdt:Binance:LinearPerpetual_bid_'

        self.btc_obs = btc_obs[['exchange_ts',
                                'receive_ts',
                                'btcusdt:Binance:LinearPerpetual_ask_price_0',
                                'btcusdt:Binance:LinearPerpetual_bid_price_0']]

        self.btc_trades = btc_trades[['exchange_ts',
                                      'receive_ts',
                                      'aggro_side',
                                      'price',
                                      'size']]

        # now - my infra will be for only 2 flows (without arrays), after - I will add them
        self.obs_ind = 0
        self.trades_ind = 0
        self.list = []

    @staticmethod
    def parse_ob(self, ob):
        return OrderbookSnapshotUpdate(
            timestamp=ob['exchange_ts'],
            ask_price=ob['btcusdt:Binance:LinearPerpetual_ask_price_0'],
            bid_price=ob['btcusdt:Binance:LinearPerpetual_bid_price_0']
        )

    @staticmethod
    def parse_trade(trade):
        return AnonTrade(
            timestamp=trade['exchange_ts'],
            side=trade['aggro_side'],
            price=trade['price'],
            size=trade['size']
        )

    def get_prepared_data(self):
        while True:
            n = self.next()
            if n == -1:
                break
            self.list.append(n)

    def next(self):
        auto_trade = False
        try:
            ob_exchange_ts = self.btc_obs.iloc[self.obs_ind]['exchange_ts']
        except:
            auto_trade = True

        auto_ob = False
        try:
            trade_exchange_ts = self.btc_trades.iloc[self.trades_ind]['exchange_ts']
        except:
            auto_ob = True

        if auto_ob and auto_trade:
            return -1
        elif auto_ob or ob_exchange_ts <= trade_exchange_ts:
            self.obs_ind += 1
            return self.parse_ob(self.btc_obs.iloc[self.obs_ind])
        elif auto_trade or ob_exchange_ts > trade_exchange_ts:
            self.trades_ind += 1
            return self.parse_trade(self.btc_trades.iloc[self.trades_ind])
        else:  # should never happen btw
            self.trades_ind += 1
            return self.parse_trade(self.btc_trades.iloc[self.trades_ind])


class Simulator:
    def __init__(self, execution_latency: float, md_latency: float):
        self.dl = DataLoader()
        self.dl.get_prepared_data()
        # self.dl.list - all market events

        self.md_queue = deque()
        self.actions_queue = deque()
        self.strategy_update_queue = deque()

        self.execution_latency = execution_latency
        self.md_latency = md_latency

        self.ask_price = 0
        self.bid_price = 0

        self.id = 0
        self.pnl = 0
        self.position = 0
        self.liq_provided = 0
        self.dl_ind = 0

    def execute_orders(self):
        """
        Тут я думаю, что по сути  мы делаем следующие итерации

        Мы получаем обновление маркет-даты (по сути 1 тик меняющий цены) и по нему исполняем или нет некоторые наши офферы.
        Поэтому мы можем просто в execute_orders передавать последний апдейт маркетдаты - и имея список открытых ордеров
        каждый раз пробегаться по нему и исполнять их все - ведь суть дека в том, что мы можем откусить первые несколько
        - а здесь нам может потребоваться удалять из середины.

        Потом если мы смотрим на ордер/его отмену (в порядке receive_ts + execution_latency) - мы обрабатываем ордера -

        если подходит по предыдущей цене или
        (подходит по новой цене и
        не отменен до новых данных (md_queue(одно событие).exchante_ts < action.timestamp + execution_latency)) -

        - он исполняется

        Пока не придумал как прописывать отмену и "оставление"
        """


        pass

    def prepare_orders(self):
        self.md_queue.append(self.dl.list[self.dl_ind])
        self.dl_ind += 1

    def place_order(self, side: str, size: float, price: float):
        order = Order(
            timestamp=time.time() * 10**6 + self.execution_latency,
            order_id=self.id,
            side=side,
            size=size,
            price=price
        )

        self.actions_queue.append(order)
        self.id += 1

    def cancel_order(self, order_id):
        # How to implement this (only O(len(queue))
        pass


    def tick(self):
        self.execute_orders()
        self.prepare_orders()

