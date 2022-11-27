from typing import List, Optional, Tuple, Union, Dict, Deque

from collections import deque, defaultdict
import numpy as np
import pandas as pd

from TechCore.Simulator.simulator import MdUpdate, Order, OwnTrade, Sim


from TechCore.Simulator.utils import get_mid_price, update_best_positions

from collections import deque


class Strategy:
    '''
        This strategy places ask and bid order every `delay` nanoseconds.
        If the order has not been executed within `hold_time` nanoseconds, it is canceled.
    '''
    def __init__(self, delay: float, risk_koef, time_oi, time_vol, avg_oi, avg_volatility, order_fees=0.00001, hold_time:Optional[float] = None) -> None:
        '''
            Args:
                delay(float): delay between orders in nanoseconds
                hold_time(Optional[float]): holding time in nanoseconds
        '''
        self.delay = delay
        if hold_time is None:
            hold_time = max( delay * 5, pd.Timedelta(10, 's').delta )
        self.hold_time = hold_time

        self.risk_koef = risk_koef
        self.normalizer = 0.001

        self.time_oi = time_oi
        self.time_vol = time_vol

        self.avg_oi = avg_oi

        self.avg_volatility = avg_volatility
        self.order_fees = order_fees
        self.atoi = []

        self.asset_position = 0
        self.usd_position = 0

        self.volatility_price_records = deque()
        self.volatility_time_records = deque()

        self.order_intensity_time_records = deque()
        self.order_intensity_size_records = deque()

        self.pnl = 0
        self.mid_price = 0
        self.total_liq = 0

        self.logs = {
            'asset_position': [],
            'usd_position': [],
            'my_spread': [],
            'stock_spread': [],
            'total_liq': [],
            'pnl': [],
            'pnl_with_liq': [],
            'order_intensity': [],
            'volatility': [],
            'midprice': [],
            'indiff_price': [],
            'own_trade_time': [],
            'place_order_time': []
        }

        self.useless_logs = []

    def run(self, sim: Sim ) -> \
            Tuple[ List[OwnTrade], List[MdUpdate], List[ Union[OwnTrade, MdUpdate] ], List[Order] ]:
        '''
            This function runs simulation

            Args:
                sim(Sim): simulator
            Returns:
                trades_list(List[OwnTrade]): list of our executed trades
                md_list(List[MdUpdate]): list of market data received by strategy
                updates_list( List[ Union[OwnTrade, MdUpdate] ] ): list of all updates
                received by strategy(market data and information about executed trades)
                all_orders(List[Orted]): list of all placed orders
        '''
        #market data list
        md_list:List[MdUpdate] = []
        #executed trades list
        trades_list:List[OwnTrade] = []
        #all updates list
        updates_list = []
        #current best positions
        best_bid = -np.inf
        best_ask = np.inf

        #last order timestamp
        prev_time = -np.inf
        #orders that have not been executed/canceled yet
        ongoing_orders: Dict[int, Order] = {}
        all_orders = []
        while True:
            #get update from simulator
            receive_ts, updates = sim.tick()
            if updates is None:
                break
            #save updates
            updates_list += updates
            for update in updates:
                #update best position
                # ----------------------------------------------------------------How to do this better-----------
                tp = "MdUpdate"

                try:
                    tmp = update.orderbook
                except:
                    try:
                        tmp = update.order_id
                        tp = "OwnTrade"
                    except:
                        tp = "Wrong"

                #print(isinstance(update, simulator.MdUpdate), isinstance(update, simulator.OwnTrade))
                # print(tp, type(update))
                if tp == "MdUpdate":
                    best_bid, best_ask = update_best_positions(best_bid, best_ask, update)
                    self.mid_price = (best_ask + best_bid) / 2

                    self.volatility_price_records.append(self.mid_price)

                    # скорее всего стоило exchange_ts, но его нет + разница слишком мала
                    self.volatility_time_records.append(update.receive_ts)

                    if update.trade is not None:
                        self.order_intensity_time_records.append(update.trade.receive_ts)
                        self.order_intensity_size_records.append(update.trade.size)

                    md_list.append(update)
                elif tp == "OwnTrade":
                    self.order_intensity_time_records.append(update.receive_ts)
                    self.order_intensity_size_records.append(update.size)

                    trades_list.append(update)
                    # delete executed trades from the dict
                    if update.order_id in ongoing_orders.keys():
                        ongoing_orders.pop(update.order_id)

                    if update.side == 'ASK':
                        self.asset_position -= update.size
                        self.usd_position += update.size * update.price
                    else:
                        self.asset_position += update.size
                        self.usd_position -= update.size * update.price

                    self.total_liq += update.size * update.price
                    self.pnl = self.asset_position * self.mid_price + self.usd_position
                    self.logs['pnl'].append(self.asset_position * self.mid_price + self.usd_position)

                    self.logs['asset_position'].append(self.asset_position)
                    self.logs['usd_position'].append(self.usd_position)
                    self.logs['total_liq'].append(self.total_liq)
                    self.logs['pnl_with_liq'].append(self.pnl + self.total_liq*self.order_fees)
                    self.logs['own_trade_time'].append(receive_ts)
                else:
                    assert False, 'Invalid type'

            if receive_ts - prev_time >= self.delay:
                prev_time = receive_ts
                mid_price = (best_bid + best_ask) / 2
                self.logs['midprice'].append(mid_price)
                self.logs['stock_spread'].append(best_ask - best_bid)

                self.volatility_price_records.append(mid_price)
                self.volatility_time_records.append(receive_ts)

                if len(self.volatility_time_records) == 0 or (self.volatility_time_records[-1] - self.volatility_time_records[0]) < self.time_vol/10:  # first iterations
                    volatility = 0
                else:
                    while self.volatility_time_records[-1] - self.volatility_time_records[0] > self.time_vol:
                        self.volatility_time_records.popleft()
                        self.volatility_price_records.popleft()

                if len(self.order_intensity_time_records) == 0 or (self.order_intensity_time_records[-1] - self.order_intensity_time_records[0]) < self.time_oi/10:  # first iterations
                    order_intensity = 1
                else:
                    while self.order_intensity_time_records[-1] - self.order_intensity_time_records[0] > self.time_oi:
                        self.order_intensity_time_records.popleft()
                        self.order_intensity_size_records.popleft()

                volatility = np.array(self.volatility_price_records).std()**2
                self.logs['volatility'].append(volatility)
                volatility /= self.avg_volatility

                order_intensity = np.array(self.order_intensity_size_records).sum() / (self.avg_oi * self.time_oi)  # can do for O(1)
                self.logs['order_intensity'].append(order_intensity)

                indifference_price = mid_price
                self.logs['indiff_price'].append(indifference_price)
                delta_x2 = self.risk_koef*volatility + 2/self.risk_koef*np.log(1 + self.risk_koef / (order_intensity + 0.01))
                self.logs['my_spread'].append(delta_x2)
                self.logs['place_order_time'].append(receive_ts)

                ask_place = indifference_price + delta_x2 / 2
                bid_place = indifference_price - delta_x2 / 2

                # place order
                bid_order = sim.place_order(receive_ts, 0.001, 'BID', bid_place)
                ask_order = sim.place_order(receive_ts, 0.001, 'ASK', ask_place)
                ongoing_orders[bid_order.order_id] = bid_order
                ongoing_orders[ask_order.order_id] = ask_order

                all_orders += [bid_order, ask_order]

            to_cancel = []
            for ID, order in ongoing_orders.items():
                if order.place_ts < receive_ts - self.hold_time:
                    sim.cancel_order( receive_ts, ID )
                    to_cancel.append(ID)
            for ID in to_cancel:
                ongoing_orders.pop(ID)

        return trades_list, md_list, updates_list, all_orders

    def get_logs(self):
        return self.logs
