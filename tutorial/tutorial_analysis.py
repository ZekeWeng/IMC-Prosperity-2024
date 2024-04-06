# Valid Libraries
import pandas as pd
import numpy as np
import statistics
import math

data = pd.read_csv("/Users/zekeweng/Desktop/for fun/IMC-Prosperity-2024/tutorial/tutorial(0)data.csv", sep=';')
df_starfruit = data[data['product'] == 'STARFRUIT'].copy()
df_amethysts = data[data['product'] == 'AMETHYSTS'].copy()

df_starfruit