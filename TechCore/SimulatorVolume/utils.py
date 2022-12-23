from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Deque, Dict

import numpy as np
from sortedcontainers import SortedDict


@dataclass
class Order:  # Our own placed order
    place_ts: float  # ts when we place the order
    exchange_ts: float  # ts when exchange(simulator) get the order
    order_id: int
    side: str
    size: float
    price: float

        
@dataclass
class CancelOrder:
    exchange_ts: float
    id_to_delete: int


@dataclass
class AnonTrade:  # Market trade
    exchange_ts: float
    receive_ts: float
    side: str
    size: float
    price: float


@dataclass
class OwnTrade:  # Execution of own placed order
    place_ts: float  # ts when we call place_order method, for debugging
    exchange_ts: float
    receive_ts: float
    trade_id: int
    order_id: int
    side: str
    size: float
    price: float
    execute: str  # BOOK or TRADE

    def __post_init__(self):
        assert isinstance(self.side, str)


@dataclass
class OrderbookSnapshotUpdate:  # Orderbook tick snapshot
    exchange_ts: float
    receive_ts: float
    asks: List[Tuple[float, float]]  # tuple[price, size]
    bids: List[Tuple[float, float]]


@dataclass
class MdUpdate:  # Data of a tick
    exchange_ts: float
    receive_ts: float
    orderbook: Optional[OrderbookSnapshotUpdate] = None
    trade: Optional[AnonTrade] = None


def update_best_positions(best_bid: float, best_ask: float, ask_quantity: float, bid_quantity: float,
                          md: MdUpdate) -> Tuple[float, float, float, float]:
    """
    Update best ask and bid prices with market data update

    :param bid_quantity:
    :param ask_quantity:
    :param best_bid:    Best bid
    :param best_ask:    Best ask
    :param md:          Market data
    :return:            New best_bid and best_ask
    """
    if md.orderbook is not None:
        best_bid = md.orderbook.bids[0][0]
        best_ask = md.orderbook.asks[0][0]
        ask_quantity = md.orderbook.asks[0][1]
        bid_quantity = md.orderbook.bids[0][1]
    elif md.trade is not None:
        if md.trade.side == 'BID':
            best_ask = max(md.trade.price, best_ask)
        elif md.trade.side == 'ASK':
            best_bid = min(md.trade.price, best_bid)
        else:
            assert False, "WRONG TRADE SIDE"
    return best_bid, best_ask, ask_quantity, bid_quantity


class PriorQueue:
    def __init__(self):
        self._queue = SortedDict()
        self._min_key = np.inf

    def push(self, key, val):
        if key not in self._queue:
            self._queue[key] = []
        self._queue[key].append(val)
        self._min_key = min(self._min_key, key)

    def pop(self):
        if len(self._queue) == 0:
            return np.inf, None
        res = self._queue.popitem(0)
        self._min_key = np.inf
        if len(self._queue):
            self._min_key = self._queue.peekitem(0)[0]
        return res

    def min_key(self):
        return self._min_key
