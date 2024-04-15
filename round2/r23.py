import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, UserId
from typing import Any, List
import numpy as np
import jsonpickle

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

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
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
        'AMETHYSTS': {"PLIMIT": 20},
        'STARFRUIT': {"PLIMIT": 20, "CACHE": []},
        'ORCHIDS':   {"PLIMIT": 100, "CONVERSION": 0}
    }

    # LR - Linear Regresssion

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}

        traderData = ""
        conversions = 0

        logger.print("traderData: " + state.traderData)
        logger.print("Observations: " + str(state.observations))

        logger.print("Market Trades: " + str(state.market_trades))
        logger.print(state.own_trades)

        for symbol in state.order_depths:

            ### Game Function
            order_depth: OrderDepth = state.order_depths[symbol]
            orders: List[Order] = []

            ### AMESTHYSTS       -       Market making + Market taking
            if symbol == "AMETHYSTS":

                ### Get ask and bid from our Market
                ask, ask_amount = list(order_depth.sell_orders.items())[0]
                bid, bid_amount = list(order_depth.buy_orders.items())[0]

                ### MM & MT around ±2
                if ask <= 9998:
                    orders.append(self.submit_order(symbol, ask, ask_amount, state))
                if bid >= 10002:
                    orders.append(self.submit_order(symbol, bid, bid_amount, state))

                orders = [x for x in orders if x is not None]

            ### STARFRUIT       -       Regression
            if symbol == "STARFRUIT":
                ### Store prices + quantities
                observed_price = np.mean([self.weighted_average_price(order_depth.buy_orders), self.weighted_average_price(order_depth.sell_orders)])
                observed_price = np.mean([list(order_depth.buy_orders.items())[0][0] , list(order_depth.sell_orders.items())[0][0]])
                logger.print("weighted" + str(np.mean(observed_price)))
                self.PRODUCTS[symbol]["CACHE"].append(observed_price)

                ### Trade on Regression Predictions
                if len(self.PRODUCTS[symbol]["CACHE"]) == 4:
                    # Weighted Linear Regression for predicted price
                    predicted_price = self.predict_regression()

                    ### Get ask and bid from our Market
                    ask, ask_amount = list(order_depth.sell_orders.items())[0]
                    bid, bid_amount = list(order_depth.buy_orders.items())[0]

                    # BUY ORDERS
                    if int(ask) < predicted_price:
                        orders.append(self.submit_order(symbol, ask, ask_amount, state))
                    # SELL ORDERS
                    if int(bid) > predicted_price:
                        orders.append(self.submit_order(symbol, bid, bid_amount, state))

                    self.PRODUCTS[symbol]["CACHE"].pop(0)

                if len(self.PRODUCTS[symbol]["CACHE"]) >= 4:
                    self.PRODUCTS[symbol]["CACHE"] == self.PRODUCTS[symbol]["CACHE"][-3]

                orders = [x for x in orders if x is not None]

            ### ORCHIDS       -       Arbitraging
            if symbol == "ORCHIDS":
                ### Get ask and bid from Southern Market (Factoring in fees)
                askThere = state.observations.conversionObservations["ORCHIDS"].askPrice + state.observations.conversionObservations["ORCHIDS"].importTariff + state.observations.conversionObservations["ORCHIDS"].transportFees
                bidThere = state.observations.conversionObservations["ORCHIDS"].bidPrice - state.observations.conversionObservations["ORCHIDS"].exportTariff - state.observations.conversionObservations["ORCHIDS"].transportFees

                ### Get ask and bid from our Market
                ask, ask_amount = list(order_depth.sell_orders.items())[0]
                bid, bid_amount = list(order_depth.buy_orders.items())[0]

                # SELL HERE BUY THERE
                if bid - askThere > 0:
                    orders.append(self.submit_order(symbol, bid, bid_amount, state))
                    if isinstance(state.position.get("ORCHIDS"), int):
                        self.PRODUCTS["ORCHIDS"]["CONVERSION"] = -state.position.get("ORCHIDS") - bid_amount

                # BUY HERE SELL THERE
                if bidThere - ask > 0:
                    orders.append(self.submit_order(symbol, ask, ask_amount, state))
                    if isinstance(state.position.get("ORCHIDS"), int):
                        self.PRODUCTS["ORCHIDS"]["CONVERSION"] = state.position.get("ORCHIDS") + ask_amount

            result[symbol] = orders

        conversions = self.PRODUCTS["ORCHIDS"]["CONVERSION"]

        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData

    ### Order Submission
    def submit_order(self, product, price, quantity, state):
        if product in state.position.keys():
            can_buy = self.PRODUCTS[product]["PLIMIT"] - state.position.get(product)
            can_sell = state.position.get(product) + self.PRODUCTS[product]["PLIMIT"]
        else:
            can_buy, can_sell = self.PRODUCTS[product]["PLIMIT"], self.PRODUCTS[product]["PLIMIT"]

        ### Fulfill Buy Orders
        if quantity < 0:
            order_quantity = min(abs(quantity), can_buy)
            logger.print("BUY", str(order_quantity) + "x", price)
            return Order(product, price, order_quantity)

        ### Fulfill Sell Orders
        if quantity > 0:
            order_quantity = min(abs(quantity), can_sell)
            logger.print("SELL", str(order_quantity) + "x", price)
            return Order(product, price, -order_quantity)

    ### Weighted Average Linear Regression
    def predict_regression(self):
        SF_coef = [0.18910402, 0.20781439, 0.26118125, 0.34190106]
        predicted_price = sum(val * SF_coef[i] for i, val in enumerate(self.PRODUCTS["STARFRUIT"]["CACHE"]))
        return predicted_price

    ### Calculate the Weighted Average
    def weighted_average_price(self, price_quantity_dict):
        total_quantity = sum(price_quantity_dict.values())
        total_value = sum(price * quantity for price, quantity in price_quantity_dict.items())
        if total_quantity == 0:
            return 0
        return total_value / total_quantity

