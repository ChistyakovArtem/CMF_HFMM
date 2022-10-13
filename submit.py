import random
from dataclasses import dataclass
from typing import Optional
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt


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
    receive_ts: float
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
class OrderbookSnapshotUpdate:  # Пока для простоты только лучшие позиции
    timestamp: float
    receive_ts: float
    ask_price: float
    bid_price: float


class DataLoader:
    def __init__(self, limit):
        self.limit = limit

        btc_obs = pd.read_csv('data/md/btcusdt_Binance_LinearPerpetual/lobs.csv')
        btc_trades = pd.read_csv('data/md/btcusdt_Binance_LinearPerpetual/trades.csv')

        btc_obs.rename(columns={' exchange_ts': 'exchange_ts'}, inplace=True)
        btc_trades.rename(columns={' exchange_ts': 'exchange_ts'}, inplace=True)

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
    def parse_ob(ob):
        return OrderbookSnapshotUpdate(
            timestamp=ob['exchange_ts'],
            receive_ts=ob['receive_ts'],
            ask_price=ob['btcusdt:Binance:LinearPerpetual_ask_price_0'],
            bid_price=ob['btcusdt:Binance:LinearPerpetual_bid_price_0']
        )

    @staticmethod
    def parse_trade(trade):
        return AnonTrade(
            timestamp=trade['exchange_ts'],
            receive_ts=trade['receive_ts'],
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
            if len(self.list) == self.limit:
                break

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
        self.dl = DataLoader(limit=1000)

        self.dl.get_prepared_data()
        # self.dl.list - all market events

        self.md_queue = deque()
        self.actions_queue = deque()
        self.strategy_update_queue = deque()

        self.execution_latency = execution_latency
        self.md_latency = md_latency

        self.ask_price = 0
        self.bid_price = 0
        self.current_time = 0

        self.logs_pnl = []
        self.logs_position = []
        self.logs_liq_provided = []
        self.logs_ask_price = []
        self.logs_bid_price = []

        self.id = 0
        self.pnl = 0
        self.position = 0
        self.liq_provided = 0
        self.dl_ind = 0

        self.first_exec = True

    def price_fit(self, side, price):
        if side == "ask":
            return price < self.ask_price
        else:
            return price > self.bid_price

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

        if self.first_exec:
            self.first_exec = False
            return

        md = self.md_queue[-1]
        try:
            self.ask_price = md.ask_price
            self.bid_price = md.bid_price
        except:
            if md.side == "ask":
                self.ask_price = min(self.ask_price, md.price)
            else:
                self.bid_price = max(self.bid_price, md.price)

        self.logs_ask_price.append(self.ask_price)
        self.logs_bid_price.append(self.bid_price)

        new_action_queue = deque()
        for act in self.actions_queue:
            # print(act.timestamp, md.timestamp, act.size, act.price, self.ask_price, self.bid_price)
            # if random.random() < 0.1:
            #     print(1/0)
            if act.timestamp > md.timestamp and self.price_fit(act.side, act.price):
                # execute

                if act.side == "ask":
                    new_position = self.position - act.size
                else:
                    new_position = self.position + act.size

                self.pnl += (self.position - new_position) * act.price  # mb ask/bid price
                self.logs_pnl.append(self.pnl)

                self.liq_provided += abs(self.position - new_position) * act.price
                self.logs_liq_provided.append(self.liq_provided)

                self.position = new_position
                self.logs_position.append(self.position)
            else:
                new_action_queue.append(act)

        self.actions_queue = new_action_queue
        self.current_time = md.timestamp

    def prepare_orders(self):
        try:
            self.md_queue.append(self.dl.list[self.dl_ind])
        except:
            return -1

        self.dl_ind += 1
        return self.dl.list[self.dl_ind - 1]

    def place_order(self, side: str, size: float, price: float):
        order = Order(
            timestamp=self.current_time + self.execution_latency*10**6,
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
        return self.prepare_orders()

    def feedback(self):
        return {
            'pnl': self.pnl,
            'liq': self.liq_provided,
            'pos': self.position,
            'pnl_with_pos': self.pnl + self.position * (self.ask_price + self.bid_price) / 2
        }

    def get_logs(self):
        return {
            'logs_pnl': self.logs_pnl,
            'logs_liq_provided': self.logs_liq_provided,
            'logs_position': self.logs_position,
            'logs_ask_price': self.logs_ask_price,
            'logs_bid_price': self.logs_bid_price
        }


class Strategy:
    def __init__(self, max_position: float) -> None:
        self.ask_price = 0
        self.bid_price = 0

    def run(self, sim: "Sim"):
        while True:
            md_update = sim.tick()
            if md_update == -1:
                break

            try:
                self.ask_price = md_update.ask_price
                self.bid_price = md_update.bid_price
            except:
                if md_update.side == "ask":
                    self.ask_price = min(self.ask_price, md_update.price)
                else:
                    self.bid_price = max(self.bid_price, md_update.price)

            if random.random() < 0.5:
                sim.place_order(
                    side="ask",
                    size=0.001,
                    price=self.ask_price
                )
            else:
                sim.place_order(
                    side="bid",
                    size=0.001,
                    price=self.bid_price
                )


if __name__ == "__main__":
    strategy = Strategy(10)
    sim = Simulator(10, 10)
    strategy.run(sim)
    print(sim.feedback())
    logs = sim.get_logs()
    #plt.plot(logs['logs_position'])
    plt.plot(logs['logs_ask_price'])
    plt.show()
