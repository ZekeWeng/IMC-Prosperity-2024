# IMC-Prosperity-2024

Athreya, Justin, Zeke



### Day 0 (4/6/24)

things we tried:
momentum
simple linear regression (update every 100 timestamps)
LR with sliding window (by batch every 50 timestamps)

momentum
poor?

with linear regression (update every 100 timestamps)
pros: both buys and sells, selling above 5000

sliding window (batch, every 100 timestamps)
positive overall returns
decent trades (sells at higher price than buys)
next: calculate sharpe ratio
batch, every 50 timestamps: improved on batches of every 100 timestamps
batch every 30 timestamps: about the same as batches of every 50 timestamps
batch every 25 timestamps: much worse than batches of 30 or 50, but sharpe ratio improved

sliding window (batch every 30 timestamps)
much better, positive sloping trend

sliding window LR (batch every 50 timestamps), on both starfruit and amethyst
best so far, reached 1.55K in profits