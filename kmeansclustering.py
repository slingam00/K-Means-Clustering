import pandas as pd
import numpy as np
import sys

'''
Description: kmeansclustering.py - In this program, I define 3 functions
# 1st function: readData() which parses the inputted file
# 2nd function: clustering() which clusters the file into a list of list of image id's
# 3rd function: calculateSC() which calculates the average Silhouette Coefficient of the clusters

All three of these functions are in a class called MyKmeans()

I also have another function called distance() which I call in order to calculate the Euclidean Distance easier
'''

class MyKmeans(): #class myKmeans
    def readData(self, filename=""): #readData function
        try:
            df = pd.read_csv(filename, header=None) #parsing the data using pandas
            return df
        except:
            return pd.DataFrame() #default return statement

    def clustering(self, parsedData = pd.DataFrame(), iterCount= 0, k= 0, centroids = None): #clustering function with default parameter values
        try:
            if iterCount < 0: #default return if the number of iterations is less than 0
                return -1
            if centroids is not None and k != len(centroids): #return an empty list if there are no centroids passed in or the k-value does not equal length of centroids
                return [[]]
            if k < 1: #default return when k is less than 1
                return -1
            if centroids is None: #when no centroids are passed in, randomize the centroids
                np.random.seed(1111)
                centroids = []
                for i in range(k):
                    centroids.append(np.random.randint(0, len(parsedData[0]))) #adding centroid numbers from dataset
            new1 = [[] for x in xrange(len(centroids))]
            y = []
            l = []
            for i in parsedData[0]:
                for j in centroids:
                    x = []
                    if i == j:
                        for k in (parsedData.ix[i][2:4]): #calculating and adding the actual centroids to a list
                            x.append(k)
                        l.append(x)
            centroids = l

            iterCount += 1
            for a in range(iterCount):
                if a == 0:
                    for i in (parsedData[0]):
                        for j in range(len(centroids)):
                            dist = 0
                            dist += distance(parsedData.ix[i, 2:4], centroids[j]) #same process except, centroids are not random
                            y.append(dist)
                        min_num = y.index(min(y))
                        new1[min_num].append(int(i))
                        y = []
                else:
                    new2 = [[] for x in xrange(len(new1))] #new nested list that calculates the new centroids after each iteration
                    for i in range(len(new1)):
                        summation1 = 0
                        summation2 = 0

                        for j in range(len(new1[i])):
                            summation1 += parsedData.ix[new1[i][j]][2]
                            summation2 += parsedData.ix[new1[i][j]][3]
                        if len(new1[i]) == 0:
                            summation1 = sys.maxint
                            summation2 = sys.maxint
                        else:
                            summation1 = float(summation1 / len(new1[i]))
                            summation2 = float(summation2 / len(new1[i]))
                        new2[i].append(summation1)
                        new2[i].append(summation2)

                    new1 = [[] for x in xrange(len(new2))]
                    y = []
                    for i in (parsedData[0]):
                        for j in range(len(new2)):
                            dist = 0
                            dist += distance(parsedData.ix[i, 2:4], new2[j])
                            y.append(dist)
                        minnum = y.index(min(y))
                        new1[minnum].append(int(i))
                        y = []
            return new1 #returns new centroids after running a certain number of iterations and clustering
        except:
            return [[]] #default return statement

    # calculate SC per image, then returns the average SC
    def calculateSC(self, clusters, parsedData = pd.DataFrame()): #calculateSC function with parameters
        try:
            sc = 0
            s = 0
            y = []
            if len(clusters[0]) == 0:
                return -1 #default return statement

            for i in range(len(clusters)): #go through each instance of clusters
                for j in range(len(clusters[i])):
                    x_val = parsedData.ix[clusters[i][j]][2]
                    y_val = parsedData.ix[clusters[i][j]][3]
                    d = 0
                    for l in range(len(clusters[i])):
                        dist1 = ((parsedData.ix[clusters[i][j]][2] - parsedData.ix[clusters[i][l]][2]) ** 2)
                        dist2 = ((parsedData.ix[clusters[i][j]][3] - parsedData.ix[clusters[i][l]][3]) ** 2)
                        eud = (dist1 + dist2) ** .5
                        d += eud
                    a = d / (len(clusters[i]) - 1)
                    b_vals = []
                    for m in range(len(clusters)):
                        if m == i:
                            continue
                        c = 0
                        for j in range(len(clusters[m])):
                            dist3 = ((x_val - parsedData.ix[clusters[m][j]][2]) ** 2)
                            dist4 = ((y_val - parsedData.ix[clusters[m][j]][3]) ** 2)
                            eu = (dist3 + dist4) ** .5
                            c += eu
                        avg = float(c / len(clusters[m]))
                        b_vals.append(avg)
                    b = min(b_vals)
                    s = float((b - a) / max(a, b)) #after calculation, s is the silhoutte coefficient for one centroid
                    sc += s #sum up all of those individual silhoutte coefficient's
            sc = sc / len(parsedData[0]) #average out
            return sc #returns the average silhoutte coefficient
        except:
            return -1 #default return

def distance(dist1, dist2): #distance function to calculate the euclidean distance easier
    x = [(a - b) ** 2 for a,b in zip(dist1,dist2)]
    y = np.sqrt(float(sum(x)))
    return y #return after calculating distance formula