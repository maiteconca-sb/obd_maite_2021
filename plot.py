import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Line Chart Example

x = np.array([0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
y = np.array([203,  533,  680,  607,  609,  924,  920, 1180,  741,  572,  603,
        714,  478,  318,  211,  481,  413,  280,  270,  312,  280,  354,
        670,  465,  329,  284,  224,  206,  229,  498,  143])

plt.close()
plt.plot(x,y)
plt.xlabel('Month')
plt.ylabel('spend (£)')
plt.show()

# 2. Bar Chart Example

plt.close()
plt.bar(x,y)
plt.xlabel('Month')
plt.ylabel('spend (£)')
plt.show()

# Plot using pandas
data = {
  'month': x,
  'spend': y*2,
  'avg': np.mean(y*2)
}

df = pd.DataFrame(data)

df.plot()
plt.show()