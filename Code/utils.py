import numpy as np
import cv2

# takes KMeans fit object (clt.fit(image)) and makes histogram of cluster colors per size of cluster
def make_hist(clt):
    clusters = clt.cluster_centers_
    labels = clt.labels_

    # label number = cluster number + 1 (to create # bins = cluster number)
    numLabels = np.arange(0, len(np.unique(labels))+1)

    # create histogram
    (hist, _) = np.histogram(labels, bins = numLabels)
    hist = hist.astype("float") # convert to float for division
    hist /= hist.sum() # divide each element by sum to get proportion

    return zip(hist, clusters)

# creates the color bar
def plot_colors(zippy):
    bar = np.zeros((50, 300, 3), dtype = "uint8") # matrix we'll fill
    startX = 0

    for (percent, color) in zippy:
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50), color.astype("uint8").tolist(), -1)
        startX = endX

    return bar

# flattens a list - general utility
def flatten(bigList):
    newList = [item for sublist in bigList for item in sublist]
    return(newList)
