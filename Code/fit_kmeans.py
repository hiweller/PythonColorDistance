from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import utils
import color_binning as ce
import cv2
import glob
import csv
import os
import variables as v

# construct argument parser so we don't have to launch python to run the script (makes bash pipelining much easier!)

# ex: $ python fish_loops.py -f Test/ -c 1 -b _1Cluster
ap = argparse.ArgumentParser()

ap.add_argument("-f", "--folder", required = True, help = "Path to folder of images (takes JPG and PNG)")

ap.add_argument("-c", "--clusters", required = True, type = int, help = "Number of clusters for KMeans sorting")

ap.add_argument("-b", "--batchname", required = False, help = "Optional identifier that will be pasted onto the end of the output files; defaults to _#Clusters")

ap.add_argument("-o", "--output", required = True, help = "Folder for putting output images and out.csv file with color cluster and percentage information")

ap.add_argument("--outcsv", required = False, help = "Optional name for output CSV; defaults to out.csv")

ap.add_argument("-s", "--saveimg", action="store_true", help = "Optional flag for storing image + color bar output images in the output folder")

ap.add_argument("-a", "--allpix", action="store_true", help = "Optional flag for fitting ALL pixels in each image instead of a subset; depending on image size could make pipeline significantly slower")

args = vars(ap.parse_args())

# get absolute filepaths to avoid redundancy issues
folder = os.path.abspath(args["folder"])
output = os.path.abspath(args["output"])
c = args["clusters"]

# if batchname wasn't specified, use default _#Clusters.png
if args["batchname"] != None:
    batch = str(args["batchname"]) + ".png"
else:
    batch = "_" + str(c) + "Clusters.png"

# if output filename is specified, use it (and make sure it ends with .csv); otherwise default to out.csv
if args["outcsv"] != None:
    if str.endswith(args["outcsv"], '.csv'):
        outCSV = output + '/' + args["outcsv"]
    else:
        outCSV = output + '/' + args["outcsv"] + '.csv'
else:
    outCSV = output + '/out.csv'


# read images from folder - read in both JPG and PNG
imageDir = glob.glob(folder + "/*.jpg")
imageDir.extend(glob.glob(folder + "/*.png"))
zippy = []

# loop through every image and get fit/output
for i in range(len(imageDir)):

    # graphical output
    if args["saveimg"]:
        # name of graphical output
        splitname = str.split(imageDir[i], '/')[-1][0:-4]

        if args["allpix"]: # full image, no graphical output
            new = ce.color_extract(imageDir[i], c, outimg=True, sample=False)
        else: # full image, graphical output
            new = ce.color_extract(imageDir[i], c, outimg=True, sample=v.sample)

        plt.savefig(output + '/' + splitname + batch)
        plt.close("all")

    # no graphical output
    else:
        # full image, no graphical output
        if args["allpix"]:
            new = ce.color_extract(imageDir[i], c, outimg=False, sample=False)
        # subset image, no graphical output (fastest)
        else:
            new = ce.color_extract(imageDir[i], c, outimg=False, sample=v.sample)

    zippy.append(new)

# formatting and generating the output CSV
colnames = ['Percent', 'R', 'G', 'B']
header = ['ID', 'Sum of Residuals', 'Avg. Pixel Distance']

for i in range(c):
    for j in range(len(colnames)):
        header.append(colnames[j] + str(i+1))

with open(outCSV, 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(zippy)
