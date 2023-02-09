from typing import List, Tuple, Union, Dict
import numpy as np
import joblib

import TechCore
import TechCore.Simulator.utils
from TechCore.Simulator.simulator import MdUpdate, Order, OwnTrade, Sim


from TechCore.Simulator.utils import update_best_positions


class Strategy:
    """
        Strategy from Stoikov's article

        This strategy places ask and bid order every `T` nanoseconds.
        If the order has not been executed within `T` nanoseconds, it is canceled.
    """
    def __init__(self, delay: float, risk_koef, min_asset_value, order_fees=-0.00004) -> None:

        """
        :param delay:                           Both "delay between orders" and "order hold" time
        :param risk_koef:                       From Stoikov article
        :param min_asset_value:                 Min asset value to scale our position (Stoikov article)
        :param order_fees:                      Market Making negative fees
        """

        self.delay = delay

        self.risk_koef = risk_koef
        self.normalizer = min_asset_value
        self.order_fees = order_fees

        self.asset_position = 0
        self.usd_position = 0

        self.pnl = 0
        self.midprice = 0
        self.total_liq = 0

        self.logs = {
            # Own Trade Logs
            'asset_position': [],       # current asset we hold

            'usd_position': [],         # current usd we hold
            'total_liq': [],            # total liquidity provided (in USD) - sum of all executed market making orders
            'pnl': [],                  # pnl = usd_position + asset_position * midprice
            'pnl_with_liq': [],         # pnl with negative commission for liquidity provided

            'own_trade_time': [],       # time records for own trades (for future plots)

            # Place Order Logs
            'best_ask': [],             # best ask price
            'best_bid': [],             # best bid price
            'midprice': [],             # (best_ask + best_bid) / 2
            'stock_spread': [],         # best_ask - best_bid = bid-ask-spread

            'ask_place': [],            # our price for ask order = indiff_price + my_spread/2
            'bid_place': [],            # our price for bid order = indiff_price - my_spread/2
            'indiff_price': [],         # "our" midprice from Stoikov's article
            'my_spread': [],            # ask_place - bid_place = deltax2

            'ask_diff': [],             # ask_place - ask_price
            'bid_diff': [],             # bid_place - bid_price

            'order_intensity': [],      # intensity of trading (order execution)
            'volatility': [],           # market volatility
            'oi_window_size': [],       # size of order intensity window (in ns)

            'place_order_time': [],     # time records for place orders (for future plots)
        }

        # ML
        self.last_orderbook = None
        self.return_predictor = joblib.load('return_predictor.pkl')
        self.volatility_predictor = joblib.load('volatility_predictor.pkl')
        self.order_intensity_predictor = joblib.load('order_intensity_predictor.pkl')

    def run(self, sim: Sim) -> \
            Tuple[List[OwnTrade], List[MdUpdate], List[Union[OwnTrade, MdUpdate]], List[Order]]:
        """
            This function runs simulation

            Args:
            Args:
                sim(Sim): simulator
            Returns:
                trades_list(List[OwnTrade]): list of our executed trades
                md_list(List[MdUpdate]): list of market data received by strategy
                updates_list( List[ Union[OwnTrade, MdUpdate] ] ): list of all updates
                received by strategy(market data and information about executed trades)
                all_orders(List[Orted]): list of all placed orders
        """
        # market data list
        md_list: List[MdUpdate] = []
        # executed trades list
        trades_list: List[OwnTrade] = []
        # all updates list
        updates_list = []
        # current best positions
        best_bid = -np.inf
        best_ask = np.inf

        # last order timestamp
        prev_time = -np.inf
        # orders that have not been executed/canceled yet
        ongoing_orders: Dict[int, Order] = {}
        all_orders = []
        while True:
            # get update from simulator
            receive_ts, updates = sim.tick()
            if updates is None:
                break
            # save updates
            updates_list += updates
            for update in updates:
                # update best position

                if isinstance(update, TechCore.Simulator.utils.MdUpdate):
                    self.last_orderbook = update.orderbook

                    best_bid, best_ask = update_best_positions(best_bid, best_ask, update)
                    self.midprice = (best_ask + best_bid) / 2

                    md_list.append(update)
                elif isinstance(update, TechCore.Simulator.utils.OwnTrade):

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
                    self.pnl = self.asset_position * self.midprice + self.usd_position

                    self.logs['pnl'].append(self.pnl)
                    self.logs['asset_position'].append(self.asset_position)
                    self.logs['usd_position'].append(self.usd_position)
                    self.logs['total_liq'].append(self.total_liq)
                    self.logs['pnl_with_liq'].append(self.pnl - self.total_liq*self.order_fees)
                    self.logs['own_trade_time'].append(receive_ts)
                else:
                    assert False, 'Invalid type'

            if receive_ts - prev_time >= self.delay:
                prev_time = receive_ts
                midprice = (best_bid + best_ask) / 2

                if self.last_orderbook is not None:
                    features = []
                    for i in range(10):
                        features.append(self.last_orderbook.asks[i][0])
                        features.append(self.last_orderbook.asks[i][1])
                        features.append(self.last_orderbook.bids[i][0])
                        features.append(self.last_orderbook.bids[i][1])

                    features = np.array(features).reshape(1, -1)

                    predicted_return = self.return_predictor.predict(features)[0]
                    predicted_volatility = self.volatility_predictor.predict(features)[0]
                    predicted_order_intensity = self.order_intensity_predictor.predict(features)[0]

                    predicted_future_price = np.exp(predicted_return) * midprice

                    indifference_price = predicted_future_price - (self.asset_position/self.normalizer)*self.risk_koef*predicted_volatility
                    self.logs['indiff_price'].append(indifference_price)

                    my_spread = self.risk_koef*predicted_volatility + 2/self.risk_koef*np.log(1 + self.risk_koef /
                                                                                              predicted_order_intensity)

                    ask_place = indifference_price + my_spread / 2
                    bid_place = indifference_price - my_spread / 2

                    self.logs['my_spread'].append(my_spread)
                    self.logs['ask_place'].append(ask_place)
                    self.logs['bid_place'].append(bid_place)
                    self.logs['ask_diff'].append(ask_place - best_ask)
                    self.logs['bid_diff'].append(bid_place - best_bid)
                    self.logs['midprice'].append(midprice)
                    self.logs['best_ask'].append(best_ask)
                    self.logs['best_bid'].append(best_bid)
                    self.logs['stock_spread'].append(best_ask - best_bid)
                    self.logs['place_order_time'].append(receive_ts)

                    # place order
                    bid_order = sim.place_order(receive_ts, 0.001, 'BID', bid_place)
                    ask_order = sim.place_order(receive_ts, 0.001, 'ASK', ask_place)
                    ongoing_orders[bid_order.order_id] = bid_order
                    ongoing_orders[ask_order.order_id] = ask_order

                    all_orders += [bid_order, ask_order]

            to_cancel = []
            for ID, order in ongoing_orders.items():
                if order.place_ts < receive_ts - self.delay:
                    sim.cancel_order(receive_ts, ID)
                    to_cancel.append(ID)
            for ID in to_cancel:
                ongoing_orders.pop(ID)

        return trades_list, md_list, updates_list, all_orders
