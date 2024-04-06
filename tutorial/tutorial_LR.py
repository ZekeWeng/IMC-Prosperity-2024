from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string


import numpy as np
import statistics
import math

class Trader:

    PRODUCTS = {
        'STARFRUIT': {"TIME": [], "DATA": [], "PRICE": 0, "LR_SIZE": 0, "PLIMIT": 0},
        'AMETHYSTS': {"TIME": [], "DATA": [], "PRICE": 0, "LR_SIZE": 0, "PLIMIT": 0}
        }

    def run(self, state: TradingState):

        POSITION_LIMITS = {'STARFRUIT': 20, 'AMETHYSTS': 20}

        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        result = {}
        for product in state.order_depths:

            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            regression_size = 50

            if len(Trader.PRODUCTS[product]["TIME"]) > regression_size:
                if Trader.PRODUCTS[product]["TIME"][-1] % (regression_size*10) == 0:
                    X = np.array(Trader.PRODUCTS[product]["TIME"][-regression_size:])
                    y = np.array(Trader.PRODUCTS[product]["DATA"][-regression_size:])

                    X = X[:, np.newaxis]
                    X = np.hstack([np.ones_like(X), X])

                    coefficients, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
                    intercept, slope = coefficients

                    Trader.PRODUCTS[product]["PRICE"] = state.timestamp * slope + intercept

            if len(order_depth.buy_orders) != 0 and len(order_depth.sell_orders) != 0:
                Trader.PRODUCTS[product]["DATA"].append(statistics.mean([list(order_depth.buy_orders.items())[0][0], list(order_depth.sell_orders.items())[0][0]]))
                Trader.PRODUCTS[product]["TIME"].append(state.timestamp)

            print("Acceptable price : " + str(Trader.PRODUCTS[product]["PRICE"]))
            print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))

            if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if int(best_ask) < Trader.PRODUCTS[product]["PRICE"]:
                    print("BUY", str(-best_ask_amount) + "x", best_ask)
                    orders.append(Order(product, best_ask, -best_ask_amount))

            if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if int(best_bid) > Trader.PRODUCTS[product]["PRICE"]:
                    print("SELL", str(best_bid_amount) + "x", best_bid)
                    orders.append(Order(product, best_bid, -best_bid_amount))

            result[product] = orders

        # String value holding Trader state data required.
        # It will be delivered as TradingState.traderData on next execution.
        traderData = "SAMPLE"

        # Sample conversion request. Check more details below.
        conversions = 1
        return result, conversions, traderData
