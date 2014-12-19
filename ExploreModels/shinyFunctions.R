#AccuracyCounts

options(gsubfn.engine="R")
library(sqldf)
library(ggplot2)
library(rgl)

load("modeltables.RData")

# More globals
MAXROWS = max(PROBLEM_DESC$nRows)
MAXN = max(PROBLEM_DESC$nTargets)
DS_RATIO = with(PROBLEM_DESC, nVars/nRows)


calculateAccuracy = function(problemfilelist, modellist = ALL_BUT_GROUND, accmat=ACCURACY_MATRIX) {
  amat = subset(accmat, Problem_File_Name %in% problemfilelist & Model_Name %in% modellist)
  sqldf("select Model_Name, avg(isRight) accuracy
        from amat
        group by Model_Name
        order by Model_Name")
}


plotAccuracy = function(acc_table, title = "Model accuracies" ) {
  best = acc_table$accuracy >= max(acc_table$accuracy)
  acc_table$isbest='no'
  acc_table$isbest[best]='yes'
  acc_table$accstring = format(acc_table$accuracy, digits=3)
  
  ggplot(acc_table, aes(x=Model_Name, y=accuracy, fill=isbest)) + 
    geom_bar(stat="identity")  + 
    scale_fill_manual(values=c("no"="darkgray", "yes"="darkblue")) + 
    geom_text(aes(y=-0.05, label=accstring)) + 
    coord_flip() + guides(fill=FALSE) + ggtitle(title)
}


# test range - 10 000 - 100 000
getAccuracyCounts = function(problemfilelist, modellist=ALL_BUT_GROUND, accmat=ACCURACY_MATRIX) {
  amat = subset(accmat, Problem_File_Name %in% problemfilelist & Model_Name %in% modellist)
  
  by_problem = sqldf("select Problem_File_Name,
                     Model_Name,
                     avg(isRight) accuracy
                     from amat
                     group by Problem_File_Name, Model_Name")
  arg_max = sqldf("select Problem_File_Name,
                  max(accuracy) maxacc
                  from by_problem
                  group by Problem_File_Name")

  most_acc = sqldf("select bp.Model_Name Model_Name,
                   count(bp.Problem_File_Name) nwins
                   from by_problem bp,
                   arg_max am
                   where bp.Problem_File_Name = am.Problem_File_Name
                   and bp.accuracy = am.maxacc
                   group by bp.Model_Name
                   order by bp.Model_Name")  
  included_models = most_acc$Model_Name
  sdiff = setdiff(modellist, included_models)
  if(length(sdiff) > 0) {
    empty = data.frame(Model_Name=sdiff, nwins=0)
    most_acc = rbind(most_acc, empty)
  }
  
  most_acc  
}

plotAccCounts = function(most_acc_table, title='Accuracy Counts') {
  best = most_acc_table$nwins >= max(most_acc_table$nwins)
  most_acc_table$isbest='no'
  most_acc_table$isbest[best]='yes'
  
  ggplot(most_acc_table, aes(x=Model_Name, y=nwins, fill=isbest)) + 
    geom_bar(stat="identity")  + 
    scale_fill_manual(values=c("no"="darkgray", "yes"="darkblue")) + 
    geom_text(aes(y=-1, label=nwins)) + 
    coord_flip() + guides(fill=FALSE) + ggtitle(title)
}


getDistanceMatrix = function(problemfilelist, modellist=ALLMODELS, distmat=DISTANCE_TABLE) {
  dmat = subset(distmat, Problem_File_Name %in% problemfilelist &
                  modelName1 %in% modellist &
                  modelName2 %in% modellist)
  distSqs = sqldf("select modelName1, 
                  modelName2,
                  sum(sqDist) sqDist
                  from dmat
                  group by modelName1, modelName2")
  as.matrix(xtabs(sqrt(sqDist) ~ modelName1+modelName2, data=distSqs))
}

distFromTruth = function(problemfilelist, modellist=ALL_BUT_GROUND, distmat=DISTANCE_TABLE) {
  dmat = subset(distmat, Problem_File_Name %in% problemfilelist &
                  modelName1 == 'ground truth' &
                  modelName2 %in% modellist)
  distSqs = sqldf("select modelName1 ground, 
                  modelName2 modelName,
                  sum(sqDist) sqDist
                  from dmat
                  group by modelName1, modelName2
                  order by modelName2")
  distSqs[, -1]  # don't need to return the column that just says ground truth
}


plotDistFromTruth = function(dist_table, title = "Model distance from ground truth" ) {
  best = which.min(dist_table$sqDist)
  dist_table$isbest='no'
  dist_table$isbest[best]='yes'
  dist_table$dist = sqrt(dist_table$sqDist)
  dist_table$diststr = format(dist_table$dist, digits=3)
  
  ggplot(dist_table, aes(x=modelName, y=dist, fill=isbest)) + 
    geom_bar(stat="identity")  + 
    scale_fill_manual(values=c("no"="darkgray", "yes"="darkblue")) + 
    geom_text(aes(y=-1.5, label=diststr)) + 
    coord_flip() + guides(fill=FALSE) + ggtitle(title)
}

plotModelDists = function(dmatrix) {
  plt3 = cmdscale(dmatrix, k=3)
  
  # set ground truth to the origin
  origin = plt3['ground truth', ] # a row
  plt3 = t(apply(plt3, 1, FUN=function(row) {row-origin}))
  if(sum(colSums(plt3)) < 0) plt3=-plt3  # reflect it so that origin is bottom left
  
  labels = LABELMAP[rownames(plt3)]
  
  rownames(plt3) = labels
  
  plot3d(plt3, size=5, col="blue", xlab="X", ylab="Y", zlab="Z")
  originrow = which(rownames(plt3)=='ground truth')
  indices = 1:dim(plt3)[1]
  for(irow in indices[-originrow]) {
    plot3d(plt3[c(irow, originrow),], add=TRUE, type="l", col="gray")
  }
  text3d(plt3, texts=rownames(plt3), adj=c(0,0)) 
  
}

plotModelDends = function(dmatrix, title="Model Groupings") {
  tree = hclust(as.dist(dmatrix))
  plot(tree, main=title, axes=FALSE, sub='', xlab='', ylab='')
}

# nclasslist = subset of c("2", "3:5", "6:10", "11:MAXN")
# ratiolist =  subset of c("c(0, 0.01)", "c(0.01, 0.03)", "c(0.03, 0.1)", "c(0.1, Inf)")
getDataSets = function(minrows, maxrows, nclasslist, ratiolist) {
  filterRows = with(PROBLEM_DESC, minrows <= nRows & nRows <= maxrows)
  
  # I feel like there ought to be an "apply" way to do this
  classnumlist = NULL
  for(str in nclasslist) {
    classnumlist = c(classnumlist, eval(parse(text=str)))
  }
  filterN = with(PROBLEM_DESC, nTargets %in% classnumlist)
  
  filterR = logical(dim(PROBLEM_DESC)[1])
 
  for(str in ratiolist) {
    interval = eval(parse(text=str))
    filterR = filterR | (interval[1] < DS_RATIO & DS_RATIO <= interval[2])
  }
  
  filter =  filterRows & filterN & filterR
  
  list(sets=PROBLEM_DESC$Problem_File_Name[filter], nsets=sum(filter))
}


#-------------------------
# print("global results")
# 
# 
# print("model accuracy")
# print(calculateAccuracy(ALLFILES))
# 
# 
# print("model distances from ground truth")
# print(distFromTruth(ALLFILES))
# 
# 
# print("how many times did each model achieve best score?")
# print(getAccuracyCounts(ALLFILES))
# 
# print("model distances")
# plotModelDists(getDistanceMatrix(ALLFILES))
# plotModelDends(getDistanceMatrix(ALLFILES), "Model grouping, all sets")
# 
