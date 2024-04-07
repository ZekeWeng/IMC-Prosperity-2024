from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import numpy as np
import statistics
import math

class Trader:

    PRODUCTS = {
        'STARFRUIT': {"QUANTITY": 0, "TIME": [], "DATA": [], "PRICE": 0, "LR_SIZE": 10, "PLIMIT": 20, "PNL": 0},
        'AMETHYSTS': {"QUANTITY": 0, "TIME": [], "DATA": [], "PRICE": 0, "LR_SIZE": 10, "PLIMIT": 20, "PNL": 0}
        }

    def run(self, state: TradingState):

        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        time = state.timestamp

        result = {}
        for product in state.order_depths:

            ### Game Function
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            ### Compute Regressions when sufficient data points
            if len(Trader.PRODUCTS[product]["TIME"]) > Trader.PRODUCTS[product]["LR_SIZE"]:
                Trader.PRODUCTS[product]["PRICE"] = Trader.compute_price_regression(product, time)

            ### Store prices + quantities
            if len(order_depth.buy_orders) != 0 and len(order_depth.sell_orders) != 0:
                Trader.PRODUCTS[product]["DATA"].append(statistics.mean([Trader.weighted_average_price(order_depth.buy_orders), Trader.weighted_average_price(order_depth.sell_orders)]))
                Trader.PRODUCTS[product]["TIME"].append(time)

            print("Median price: " + str(statistics.mean([list(order_depth.buy_orders.items())[0][0], list(order_depth.sell_orders.items())[0][0]])))
            print("WA price: " + str(Trader.weighted_average_price(order_depth.buy_orders)))

            print("Acceptable price : " + str(Trader.PRODUCTS[product]["PRICE"]))
            print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))

            ### BUY ORDERS
            if len(order_depth.sell_orders) != 0:
                ask, ask_amount = list(order_depth.sell_orders.items())[0]
                if int(ask) < Trader.PRODUCTS[product]["PRICE"] and time > Trader.PRODUCTS[product]["LR_SIZE"] * 100:
                    order = Trader.submit_order(product, "BUY", ask, ask_amount)
                    orders.append(order)

            ### SELL ORDERS
            if len(order_depth.buy_orders) != 0:
                bid, bid_amount = list(order_depth.buy_orders.items())[0]
                if int(bid) > Trader.PRODUCTS[product]["PRICE"] and time > Trader.PRODUCTS[product]["LR_SIZE"] * 100:
                    order = Trader.submit_order(product, "SELL", bid, bid_amount)
                    orders.append(order)

            result[product] = orders

        # String value holding Trader state data required.
        # It will be delivered as TradingState.traderData on next execution.
        traderData = "SAMPLE" # ID TEAM???

        # Sample conversion request. Check more details below.
        # FIGURE THIS OUT LATER
        conversions = 1
        return result, conversions, traderData

    def submit_order(product, action, price, quantity):
        order_quantity = min(abs(quantity), Trader.PRODUCTS[product]["PLIMIT"])
        print(action, str(order_quantity) + "x", price)
        if action == "BUY":
            return Order(product, price, order_quantity)
        if action == "SELL":
            return Order(product, price, -order_quantity)

    def compute_price_regression(product, time):
        X = np.array(Trader.PRODUCTS[product]["TIME"][-Trader.PRODUCTS[product]["LR_SIZE"]:])
        y = np.array(Trader.PRODUCTS[product]["DATA"][-Trader.PRODUCTS[product]["LR_SIZE"]:])

        X = X[:, np.newaxis]
        X = np.hstack([np.ones_like(X), X])

        coefficients, _residuals, _rank, _s = np.linalg.lstsq(X, y, rcond=None)
        intercept, slope = coefficients

        return time * slope + intercept

    def compute_momentum():
        pass

    def weighted_average_price(price_quantity_dict):
        total_quantity = sum(price_quantity_dict.values())
        total_value = sum(price * quantity for price, quantity in price_quantity_dict.items())

        if total_quantity == 0:
            return 0

        return total_value / total_quantity