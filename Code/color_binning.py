from sklearn.cluster import KMeans # for KMeans fitting
from sklearn.utils import shuffle # for shuffling pixels before random selection
import matplotlib.pyplot as plt # for plotting histograms
import argparse # for passing arguments from bash
import utils # must be in the SAME DIRECTORY
import cv2
import numpy as np
import variables as v

# uses KMeans clustering to find mean color clusters
# takes image path and cluster number
# optional: if outimg=True, then will also return a plot with original image + cluster histogram
# optional: if sample is a number (isinstance(sample, int)==True), then function randomly samples that number of pixels from the image
# recommend sampling ~10,000-30,000 pixels - much faster than clustering 100% of pixels if you have large images!

def color_extract(path, clusters, outimg=False, sample=False):

    lower = v.lower_range
    upper = v.upper_range

    image = cv2.imread(path) # read in raw image (BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert to RGB
    pix = image.reshape(image.shape[0]*image.shape[1],3) # reshape into a 3-column matrix

    mask = cv2.inRange(image, lower, upper) # index pixels in the 'green' range (go outside of just [0, 255, 1] since pixels on edges tend to get compressed oddly)
    # in this mask, we're indexing anything IN THE GREEN range - i.e., we only want to keep things that have an index value of 0

    mask = mask.reshape(mask.shape[0]*mask.shape[1],) # reshape into a vector for indexing

    pix = pix[mask==0,] # no more green pixels!

    # if sample is a number of pixels, then randomly select that number of pixels from this count
    if isinstance(sample, int):
        pix = shuffle(pix, random_state=0)[:sample]

    # perform actual KMeans fitting:
    clt = KMeans(n_clusters = clusters)
    clt.fit(pix)

    zippy = utils.make_hist(clt)

    # average distance from pixel to its assigned cluster
    meanDist = clt.inertia_/pix.shape[0]

    # if outimg = true, make the corresponding plot
    if outimg:

        bar = utils.plot_colors(zippy) # color bar

        fig, (ax1, ax2) = plt.subplots(2, 1) # initialize plot

        im1 = ax1.imshow(image) # original image on top
        im2 = ax2.imshow(bar) # color bar on bottom

        # turns off axis ticks/labels, etc
        ax1.axes.get_xaxis().set_visible(False)
        ax1.axes.get_yaxis().set_visible(False)
        ax2.axes.get_xaxis().set_visible(False)
        ax2.axes.get_yaxis().set_visible(False)
        fig.tight_layout()
        plt.axis("off")

    # build zippy object - clusters (in RGB), mean pixel distance, sum of residuals, image path
    zippy = [i for j in zippy for i in j]
    zippy = [i for j in zippy for i in j.flatten('F')]
    zippy = [meanDist] + zippy
    zippy = [clt.inertia_] + zippy
    zippy.insert(0, path)

    # returns zip object
    return zippy
