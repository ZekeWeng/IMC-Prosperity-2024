from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

class Trader:

    POSITION_LIMITS = {'STARFRUIT' : 20, 'AMETHYSTS' : 20}

    previous_bid = 10000
    previous_ask = -10000

    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

		# Orders to be placed on exchange matching engine
        result = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            # print("Acceptable price : " + str(acceptable_price))
            print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))

            if product == "STARFRUIT":
                if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
                    best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                    best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]

                    if best_bid > Trader.previous_bid:
                        print("BUY", str(-best_ask_amount) + "x", best_ask)
                        orders.append(Order(product, best_ask, -best_ask_amount))

                    if best_ask < Trader.previous_ask:
                        print("SELL", str(best_bid_amount) + "x", best_bid)
                        orders.append(Order(product, best_bid, -best_bid_amount))

                Trader.previous_bid = best_bid
                Trader.previous_ask = best_ask

            result[product] = orders

        # String value holding Trader state data required.
        # It will be delivered as TradingState.traderData on next execution.
        traderData = "SAMPLE"

        # Sample conversion request. Check more details below.
        conversions = 1
        return result, conversions, traderData
