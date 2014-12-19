#!/usr/bin/python

# script from Win-Vector LLC http://www.win-vector.com/ GPL3 license
# run the data from through scikit-learn
#   http://jmlr.csail.mit.edu/papers/v15/delgado14a.html
# "Do we Need Hundreds of Classifiers to Solve Real World Classification Problems?" Manuel Fernandez-Delgado, Eva Cernadas, Senen Barro, Dinani Amorim; 15(Oct):3133-3181, 2014
# see http://www.win-vector.com/blog/2014/12/a-comment-on-preparing-data-for-classifiers/ (and other blog articles) for some details
# reads argv[1] which should be a gzipped tar file containing arff files ending in .arrf
# expects all input variables to be numeric and result variable to be categorical
# produces model modelres.csv.gz and problemDescr.csv
# R code to look at accuracy:
#   m <- read.table('modelres.csv.gz',header=TRUE,sep=',')
#   aggregate(ArgMaxIndicator~Problem.File.Name+Model.Name,data=subset(m, PredictionIndex== Correct.Answer),FUN=mean)


import re
import math
import tarfile
import random
import numpy.random
import csv
import sys
import time
import os
# liac-arff https://pypi.python.org/pypi/liac-arff 'pip install liac-arff'
import arff 
import gzip
import pandas
import numpy
import sklearn.svm
import sklearn.ensemble
import sklearn.tree
import sklearn.naive_bayes
import sklearn.linear_model
import sklearn.neighbors
import scipy.optimize
import sklearn.neural_network
import multiprocessing
import contextlib

print 'start',os.getpid(),time.strftime("%c")
sys.stdout.flush()


testTarget = 100
testStrategyCut = 500
doXForm = False
maxXFormTrainSize = 5000
maxCTrainSize = 200000


randSeed = 2382085237
random.seed(randSeed)
numpy.random.seed(randSeed)



# based on http://scikit-learn.org/stable/auto_examples/plot_rbm_logistic_classification.html#example-plot-rbm-logistic-classification-py
class RBMform:
   """"Restricted Boltzman / transform"""
   def __init__(self, random_state):
      self.xform_ = sklearn.neural_network.BernoulliRBM(random_state=random_state)
   def scale_(self,x):
      nRow = len(x)
      def f(i,j):
         return min(1.0,max(0.0,(x[i][j]-self.colrange_[j][0])/(1.0*(self.colrange_[j][1]-self.colrange_[j][0]))))
      return [ [ f(i,j) for j in self.varIndices_ ] for i in range(nRow) ]
   def fit(self, xTrain):
      # rescale to range [0,1]
      xTrain = numpy.array(xTrain)
      nCol = len(xTrain[0])
      self.colrange_ = [ [min(xTrain[:,j]),max(xTrain[:,j])] for j in range(nCol)]
      self.varIndices_ = [ j for j in range(nCol) if self.colrange_[j][1]>(self.colrange_[j][0]+1.0e-6) ]
      xScale = self.scale_(xTrain)
      self.xform_.fit(xScale)
   def transform(self,xTest):
      nRow = len(xTest)
      sc = self.scale_(xTest)
      xf = self.xform_.transform(self.scale_(xTest))  
      # return transformed plus original scaled columns
      return [ numpy.array(sc[i]).tolist() + numpy.array(xf[i]).tolist() for i in range(nRow) ]


# map of modelfactories (each takes a random state on instantiation)
model_dict = {"SVM" : (lambda rs: sklearn.svm.SVC(kernel='rbf',gamma=0.001,C=10.0,probability=True,random_state=rs)),\
              "Random Forest" : (lambda rs: sklearn.ensemble.RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1,random_state=rs)),\
              "Gradient Boosting" : (lambda rs: sklearn.ensemble.GradientBoostingClassifier(random_state=rs)), \
              "Decision Tree" : (lambda rs: sklearn.tree.DecisionTreeClassifier(random_state=rs)), \
              "Naive Bayes" : (lambda rs: sklearn.naive_bayes.GaussianNB()), \
              "Logistic Regression" : (lambda rs: sklearn.linear_model.LogisticRegression(random_state=rs)), \
              "KNN5" : (lambda rs: sklearn.neighbors.KNeighborsClassifier(n_neighbors=5))}



tarFileName = sys.argv[1] # data.tar.gz
doutfile = 'modelres.csv.gz'
poutfile = 'problemDescr.csv'
print 'reading:',tarFileName
print 'writing data to:',doutfile
print 'writing problem summaries to:',poutfile
sys.stdout.flush()
realTypes = set(['REAL', 'NUMERIC'])

# mark first maximal element with 1, else with zero
# v numeric array
def argMaxV(v):
   n = len(v)
   maxV = max(v)
   mv = numpy.zeros(n)
   for j in range(n):
      if v[j]>=maxV:
         mv[j] = 1.0
         break
   return mv

# raise all entries up to a common floor while preserving total
# this is a smoothing term (replacing things near zero with a floor, and scaling other entries to get mass)
# and the idea is a classifier already staying away from zero is unaffected
# v numeric array of non-negative entries summing to more than epsilon*len(v)
# example smoothElevate([0.5,0.39,0.11,0.0],0.1) -> [0.44943820224719105, 0.35056179775280905, 0.1, 0.1]
def smoothElevate(vIn,epsilon):
   v = numpy.array(vIn)
   n = len(v)
   while True:
      peggedIndices = set([i for i in range(n) if v[i]<=epsilon])
      neededMass = sum([epsilon - v[i] for i in peggedIndices])
      if neededMass<=0.0:
         return v
      else:
         scale = (1.0 - len(peggedIndices)*epsilon)/sum([vIn[i] for i in range(n) if i not in peggedIndices])
         v = [ epsilon if i in peggedIndices else scale*vIn[i] for i in range(n) ]

def processArff(name,reln):
   rname = reln['relation']
   attr = reln['attributes']
   print 'start\t',rname,os.getpid(),time.strftime("%c")
   sys.stdout.flush()
   data = reln['data']
   ncol = len(attr)
   # assuming unique non-numeric feature is category to be predicted
   features = [ i for i in range(ncol) if isinstance(attr[i][1],basestring) and attr[i][1] in realTypes ]
   outcomes = [ i for i in range(ncol) if not i in set(features) ]
   if (not len(outcomes)==1) or (len(features)<=0):
      print '\tskip (not in std form)',name,attr,os.getpid(),time.strftime("%c")
      sys.stdout.flush()
      return None
   outcome = outcomes[0]
   xvars = [ [ row[i] for i in features ] for row in data ]
   yvar = [ row[outcome] for row in data ]
   nrow = len(yvar)
   levels = sorted([ l for l in set(yvar) ])
   nlevels = len(levels)
   levelMap = dict([ (levels[j],j) for j in range(nlevels) ])
   prng = numpy.random.RandomState(randSeed)
   def scoreRows(testIndices):
      scoredRowsI = []
      trainIndices = [ i for i in range(nrow) if i not in set(testIndices) ]
      if len(trainIndices)>maxCTrainSize:
         trainIndices = sorted(prng.choice(trainIndices,size=maxCTrainSize,replace=False))
      xTrain = [ xvars[i] for i in trainIndices ]
      yTrain = [ yvar[i] for i in trainIndices ]
      nTrain = len(yTrain)
      xTest = [ xvars[i] for i in testIndices ]
      yTest = [ yvar[i] for i in testIndices ]
      nTest = len(yTest)
      trainLevels = sorted([ l for l in set(yTrain) ])
      if (nTrain>0) and (nTest>0) and (len(trainLevels)==nlevels):
         # naive non-parametric empirical idea: 
         #    no evidence any event can be rarer than 1/(nTrain+1)
         #    the elevate/smoothing code needs the total weight
         #    to sum to 1 and to have nlevels*epsilon < 1 so there is
         #    weight to move around (so we add the 1/(1+nlevels) condition)
         epsilon = min(1.0/(1.0+nlevels),1.0/(1.0+nTrain)) 
         for i in range(nTest):
            ti = numpy.zeros(nlevels)
            ti[levelMap[yTest[i]]] = 1
            adjTruth = smoothElevate(ti,epsilon)
            for j in range(nlevels):
               scoredRowsI.append([name,'ground truth',testIndices[i],levelMap[yTest[i]],j,ti[j],ti[j],adjTruth[j]])
         def runModels(mSuffix,xTrainD,xTestD):
            for modelName, modelFactory in model_dict.iteritems():
               model = modelFactory(numpy.random.RandomState(randSeed))
               model.fit(xTrainD, yTrain)
               assert len(model.classes_)==nlevels
               for j in range(nlevels):
                  assert levels[j]==model.classes_[j]
               pred = model.predict_proba(xTestD)
               for i in range(nTest):
                  sawMax = False
                  maxIndicator = argMaxV(pred[i])
                  adjPred = smoothElevate(pred[i],epsilon)
                  for j in range(nlevels):
                      scoredRowsI.append([name,modelName+mSuffix,testIndices[i],levelMap[yTest[i]],j,pred[i][j],maxIndicator[j],adjPred[j]])
         runModels('',xTrain,xTest)
         if doXForm:
            xform = RBMform(numpy.random.RandomState(randSeed))
            if nTrain<=maxXFormTrainSize:
               xform.fit(xTrain)
            else:
               xIndices = sorted(prng.choice(range(nTrain),size=maxXFormTrainSize,replace=False))
               xform.fit([xTrain[i] for i in xIndices])
            runModels(' (RBM)',xform.transform(xTrain),xform.transform(xTest))
      return scoredRowsI
   print 'initial \t',[name,rname,nrow,len(features),nlevels],os.getpid(),time.strftime("%c")
   sys.stdout.flush()
   scoredRows = []
   nTested = 0
   if (nrow>10) and (nlevels>1):
      if nrow>testStrategyCut:
         # single hold-out testing
         nTries = 10
         while (nTested<=0) and (nTries>0):
            testIndices = sorted(prng.choice(range(nrow),size=testTarget,replace=False))
            nTries -= 1
            sri = scoreRows(testIndices)
            if len(sri)>0:
               scoredRows.extend(scoreRows(testIndices))
               nTested = testTarget
         probDescr = [name,rname,nrow,'test-set cross-validation',nrow-testTarget,nTested,len(features),nlevels]
      else:
         # hold out one cross validation
         candidates = [ i for i in range(nrow) ]
         prng.shuffle(candidates)
         for ci in candidates:
            testIndices = [ ci ]
            sri = scoreRows(testIndices)
            if len(sri)>0:
               scoredRows.extend(sri)
               nTested += 1
               if nTested>=testTarget:
                  break
         probDescr = [name,rname,nrow,'leave-one-out cross-validation',nrow-1,nTested,len(features),nlevels]
      print '\tdone\t',probDescr,os.getpid(),time.strftime("%c")
      sys.stdout.flush()
      if nTested>0:
         return { 'pDesc': probDescr, 'scoredRows': scoredRows }
   else:
      print '\tskip (could not split)',name,os.getpid(),time.strftime("%c")
      sys.stdout.flush()
   return None


def processArffByName(targetName):
   with contextlib.closing(tarfile.open(tarFileName,'r:gz')) as t_in:
      for member in t_in.getmembers():      
         name = member.name
         if targetName==name:
            return processArff(name,arff.load(t_in.extractfile(member)))
   return None         

# get targets
names = []
with contextlib.closing(tarfile.open(tarFileName,'r:gz')) as t_in:
   for member in t_in.getmembers():      
      name = member.name
      if not name.endswith('.arff'):
         continue
      sys.stdout.flush()
      if name.endswith('_test.arff'):
         print '\tskip (test)',name
         sys.stdout.flush()
         continue
      names.append(name)

print names,os.getpid(),time.strftime("%c")
pool = multiprocessing.Pool()
results = pool.map(processArffByName,names)
print 'done processing pool',os.getpid(),time.strftime("%c")

# write out results
with contextlib.closing(gzip.open(doutfile,'wb')) as f_out:
   with open(poutfile,'wb') as p_out:
      pwriter = csv.writer(p_out,delimiter=',')
      pwriter.writerow(['Problem File Name','Relation Name',"nRows","TestMethod","nTrain","nTest","nVars","nTargets"])
      dwriter = csv.writer(f_out,delimiter=',')
      dwriter.writerow(['Problem File Name','Model Name','Row Number','Correct Answer','PredictionIndex','ProbIndex','ArgMaxIndicator','SmoothedProbIndex'])
      for res in results:
         if res is not None:
            pwriter.writerow(res['pDesc'])
            for r in res['scoredRows']:
               dwriter.writerow(r)
         f_out.flush()
         p_out.flush()
            
print 'done',os.getpid(),time.strftime("%c")
sys.stdout.flush()
