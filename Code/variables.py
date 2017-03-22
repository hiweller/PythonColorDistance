import numpy as np

### BACKGROUND COLOR FOR REMOVAL ###
bg_col = np.array([0, 255, 1], dtype="uint8")
lower_range = np.array([0,100,0], dtype=np.uint8)
upper_range = np.array([80,255,80], dtype=np.uint8)

### NUMBER OF PIXELS FOR RANDOM SAMPLING ###
sample = 10000
