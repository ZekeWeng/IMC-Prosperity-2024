import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, UserId
from typing import Any, List
import numpy as np

class Trader:

    PRODUCTS = {
        'AMETHYSTS': {"TIME": [], "DATA": [], "DELTAS": [0], "PRICE": 0, "PLIMIT": 20, "STRATEGY": "LR", "LR_SIZE": 9},
        'STARFRUIT': {"TIME": [], "DATA": [], "DELTAS": [0], "PRICE": 0, "PLIMIT": 20, "STRATEGY": "LR", "LR_SIZE": 9}
    }

    # LR - Linear Regresssion

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}

        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        for symbol in state.order_depths:

            ### Game Function
            order_depth: OrderDepth = state.order_depths[symbol]
            orders: List[Order] = []

            ### Store prices + quantities
            if len(order_depth.buy_orders) != 0 and len(order_depth.sell_orders) != 0:
                observed_price = np.mean([self.weighted_average_price(order_depth.buy_orders), self.weighted_average_price(order_depth.sell_orders)])
                if len(self.PRODUCTS[symbol]["DATA"]) > 0:
                    self.PRODUCTS[symbol]["DELTAS"].append(self.PRODUCTS[symbol]["DATA"][-1] - observed_price)
                self.PRODUCTS[symbol]["DATA"].append(observed_price)
                self.PRODUCTS[symbol]["TIME"].append(state.timestamp)

            ### Compute Regressions when sufficient data points
            if len(self.PRODUCTS[symbol]["TIME"]) > self.PRODUCTS[symbol]["LR_SIZE"]:
                self.PRODUCTS[symbol]["PRICE"] = self.compute_price_regression(symbol)

            if self.PRODUCTS[symbol]["STRATEGY"] == "LR" and state.timestamp > self.PRODUCTS[symbol]["LR_SIZE"] * 100:
                ### Buy + Sell Best Price
                # BUY ORDERS
                if len(order_depth.sell_orders) != 0:
                    ask, ask_amount = list({k: order_depth.sell_orders[k] for k in sorted(order_depth.sell_orders)}.items())[0]
                    if int(ask) < self.PRODUCTS[symbol]["PRICE"]:
                        orders.append(self.submit_order(symbol, ask, ask_amount, state))
                # SELL ORDERS
                if len(order_depth.buy_orders) != 0:
                    bid, bid_amount = list({k: order_depth.buy_orders[k] for k in sorted(order_depth.buy_orders, reverse=True)}.items())[0]
                    if int(bid) > self.PRODUCTS[symbol]["PRICE"]:
                        orders.append(self.submit_order(symbol, bid, bid_amount, state))
                orders = [x for x in orders if x is not None]

            result[symbol] = orders

        traderData = ""
        conversions = 1

        return result, conversions, traderData

    def submit_order(self, product, price, quantity, state):
        if product in state.position.keys():
            can_buy = self.PRODUCTS[product]["PLIMIT"] - state.position.get(product)
            can_sell = state.position.get(product) + self.PRODUCTS[product]["PLIMIT"]
        else:
            can_buy, can_sell = self.PRODUCTS[product]["PLIMIT"], self.PRODUCTS[product]["PLIMIT"]
        if quantity < 0:                                                                            # BUY ORDERS
            order_quantity = min(abs(quantity), can_buy)
            print("BUY", str(order_quantity) + "x", price)
            return Order(product, price, order_quantity)
        if quantity > 0:                                                                            # SELL ORDERS
            order_quantity = min(abs(quantity), can_sell)
            print("SELL", str(order_quantity) + "x", price)
            return Order(product, price, -order_quantity)

    def compute_price_regression(self, product):
        X = np.array(self.PRODUCTS[product]["TIME"][-self.PRODUCTS[product]["LR_SIZE"]:])
        y = np.array(self.PRODUCTS[product]["DATA"][-self.PRODUCTS[product]["LR_SIZE"]:])

        X = X[:, np.newaxis]
        X = np.hstack([np.ones_like(X), X])

        coefficients, _residuals, _rank, _s = np.linalg.lstsq(X, y, rcond=None)
        intercept, slope = coefficients

        return (self.PRODUCTS[product]["TIME"][-1] + 100) * slope + intercept

    def weighted_average_price(self, price_quantity_dict):
        total_quantity = sum(price_quantity_dict.values())
        total_value = sum(price * quantity for price, quantity in price_quantity_dict.items())
        if total_quantity == 0:
            return 0
        return total_value / total_quantity