#!/bin/bash

# provide max and min number of clusters, step size, and output folder

cStart=10 # min cluster number
cEnd=50 # max cluster number
s=10 # step size (from cStart to cEnd in steps of s)


# folder full of images you want to analyze
input=/Users/hannah/Dropbox/Westneat_Lab/ColorDistance/Images/

output=/Users/hannah/Dropbox/Westneat_Lab/ColorDistance/Output/

for i in $(seq $cStart $s $cEnd)
do
	mkdir -p "${output}${i}Clusters"
	outfolder=${output}${i}Clusters/
	python /Users/hannah/Dropbox/Westneat_Lab/ColorDistance/Code/fit_kmeans.py -f ${input} -o ${outfolder} -c ${i} -s
done
