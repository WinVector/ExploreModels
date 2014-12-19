# setwd("~/Documents/Projects/scratch.git/shapeOfML")

options(gsubfn.engine="R")
library(sqldf)


#
# Globals
#
PROBLEM_DESC = read.table('data/problemDescr.csv', header=TRUE, sep=',', stringsAsFactors=TRUE)
colnames(PROBLEM_DESC) = gsub('\\.', '_', colnames(PROBLEM_DESC) )

MODEL_RESULTS = read.table('data/modelres.csv.gz', header=TRUE, sep=',', stringsAsFactors=TRUE)
colnames(MODEL_RESULTS) = gsub('\\.', '_', colnames(MODEL_RESULTS) )
# remove the RBM
# isRBM = grepl("(RBM)", MODEL_RESULTS$Model_Name)
# MODEL_RESULTS = subset(MODEL_RESULTS, !isRBM)
# MODEL_RESULTS$Model_Name = factor(MODEL_RESULTS$Model_Name)


ALLMODELS = sort(unique(MODEL_RESULTS$Model_Name))     

ALL_BUT_GROUND = setdiff(ALLMODELS, c('ground truth'))

LABELMAP = c("DT", "GB", "ground truth", "KNN5", "LR", "NB", "RF", "SVM")
names(LABELMAP) = ALLMODELS

ALLFILES = MODEL_RESULTS$Problem_File_Name  


#
# Functions
#

#
# Just calculate the accuracy matrix once; we can use subsets of it later
# [Problem_File_Name, Model_Name, Row_Number, PredictionIndex, Correct_Answer, isRight]
#
calculateAccuracyMatrix = function(modelres=MODEL_RESULTS) {
  # first, get the maximum probabilities --- change this to use ArgMaxIndicator column
  
  maxProbs = sqldf("select * from modelres
                   where ArgMaxIndicator=1
                   group by Problem_File_Name, Model_Name, Row_Number
                   order by Problem_File_Name, Model_Name, Row_Number")
  maxProbs$isRight = as.numeric(with(maxProbs,PredictionIndex==Correct_Answer))
  maxProbs
}

ACCURACY_MATRIX = calculateAccuracyMatrix()

#
# calculate the matrix [Problem_File_Name, modelName1, modelName2, sqDist]
#
#  NOTE: if I do a similar calculation with deviance, use SmoothedProbIndex column
#
calculateDistanceTable = function(modelres=MODEL_RESULTS) {
  distSqs = sqldf("select 
                  m1.Problem_File_Name Problem_File_Name,
                  m1.Model_Name modelName1,
                  m2.Model_Name modelName2,
                  sum(square(m1.ProbIndex-m2.ProbIndex)) sqDist
                  from 
                  modelres m1,
                  modelres m2
                  where
                  m1.Problem_File_Name = m2.Problem_File_Name
                  and m1.Row_Number = m2.Row_Number
                  and m1.PredictionIndex = m2.PredictionIndex
                  group by
                  m1.Problem_File_Name,
                  m1.Model_Name,
                  m2.Model_Name
                  order by m1.Problem_File_Name, m1.Model_Name, m2.Model_Name")
  distSqs
}

DISTANCE_TABLE = calculateDistanceTable()  

save(PROBLEM_DESC, MODEL_RESULTS, ALLMODELS, ALL_BUT_GROUND, LABELMAP, 
     ALLFILES,ACCURACY_MATRIX, DISTANCE_TABLE, 
     file="ShinyApps/modeltables.RData")
