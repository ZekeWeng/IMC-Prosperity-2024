import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, UserId
from typing import Any, List
import string
import numpy as np
import statistics
import math

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[:max_length - 3] + "..."

logger = Logger()

class Trader:

    PRODUCTS = {
        'AMETHYSTS': {"TIME": [], "DATA": [], "DELTAS": [0], "PRICE": 0, "PLIMIT": 20, "STRATEGY": "LR", "LR_SIZE": 9},
        'STARFRUIT': {"TIME": [], "DATA": [], "DELTAS": [0], "PRICE": 0, "PLIMIT": 20, "STRATEGY": "LR", "LR_SIZE": 9}
    }

    # LR - Linear Regresssion

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}

        logger.print("traderData: " + state.traderData)
        logger.print("Observations: " + str(state.observations))

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

            # logger.print(state.position.get(symbol))
            # print(state.position.get(symbol))
            # logger.print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))
            # print("Current Strategy: " + str(self.PRODUCTS[symbol]["STRATEGY"]))
            # print("Current Pred Price: " + str(self.PRODUCTS[symbol]["PRICE"]))


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

            # if symbol in state.position.keys():
            #     if state.position[symbol]/self.PRODUCTS[symbol]["PLIMIT"] >= 0.5:
            #         if len(order_depth.buy_orders) != 0:
            #             bid, bid_amount = list(order_depth.buy_orders.items())[0]
            #             if int(bid) > self.PRODUCTS[symbol]["PRICE"] - .75:
            #                 orders.append(self.submit_order(symbol, bid, bid_amount, state))
            #     if state.position[symbol]/self.PRODUCTS[symbol]["PLIMIT"] <= -0.5:
            #         if len(order_depth.sell_orders) != 0:
            #             ask, ask_amount = list(order_depth.sell_orders.items())[0]
            #             if int(ask) < self.PRODUCTS[symbol]["PRICE"] + .25:
            #                 orders.append(self.submit_order(symbol, ask, ask_amount, state))

            #     orders = [x for x in orders if x is not None]

            # if self.PRODUCTS[symbol]["STRATEGY"] == "LR" and state.timestamp > self.PRODUCTS[symbol]["LR_SIZE"] * 100:
            #     orders.extend(self.execute(symbol, order_depth, state))
            #     orders = [x for x in orders if x is not None]

            result[symbol] = orders

        traderData = ""
        conversions = 1

        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData

    def execute(self, product, order_depth, state):                                                 # SELL EVERYTHING
        orders = []
        for depth in range(0, len(order_depth.buy_orders)):
            bid, bid_amount = list(order_depth.buy_orders.items())[depth]
            if int(bid) > self.PRODUCTS[product]["PRICE"]:
                orders.append(self.submit_order(product, bid, bid_amount, state))
            else:
                break
        for depth in range(0, len(order_depth.sell_orders)):
            ask, ask_amount = list(order_depth.sell_orders.items())[depth]
            if int(ask) < self.PRODUCTS[product]["PRICE"]:
                orders.append(self.submit_order(product, ask, ask_amount, state))
            else:
                break
        return orders

    def submit_order(self, product, price, quantity, state):
        if product in state.position.keys():
            can_buy = self.PRODUCTS[product]["PLIMIT"] - state.position.get(product)
            can_sell = state.position.get(product) + self.PRODUCTS[product]["PLIMIT"]
        else:
            can_buy, can_sell = self.PRODUCTS[product]["PLIMIT"], self.PRODUCTS[product]["PLIMIT"]
        if quantity < 0:                                                                            # BUY ORDERS
            order_quantity = min(abs(quantity), can_buy)
            logger.print("BUY", str(order_quantity) + "x", price)
            return Order(product, price, order_quantity)
        if quantity > 0:                                                                            # SELL ORDERS
            order_quantity = min(abs(quantity), can_sell)
            logger.print("SELL", str(order_quantity) + "x", price)
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