#AccuracyCounts

#
# from directory above this one:
# library(shiny)
# runApp("ExploreModels") 
#
options(rgl.useNULL=TRUE)
library(shiny)
source("shinyFunctions.R")


shinyServer(function(input, output) {
  
  dataSets = reactive (
    getDataSets(input$minrows, input$maxrows, input$nclasses, input$ratio)
  )
  
  accTable = reactive (
    getAccuracyCounts(dataSets()$sets)
  )
  
  avgAccTable = reactive (
    calculateAccuracy(dataSets()$sets)
  )
  
  dmatrix = reactive (
    getDistanceMatrix(dataSets()$sets)
  )
  
  # Average Accuracy tab
  output$plotlabelAA = renderText(paste("Average model accuracy over sets with", 
                                        input$minrows, "to", input$maxrows, "rows"))
  
  output$accPlot = renderPlot( {
    title = paste(dataSets()$nsets, "sets in total")
    plotAccuracy(avgAccTable(), title)
  })
  
  # Accuracy counts tab
  
  output$plotlabelAC = renderText(paste("How many times is each model maximally accurate for data sets of size", 
                                      input$minrows, "to", input$maxrows, "rows?"))
  
  output$countPlot <- renderPlot({
    title = paste(dataSets()$nsets, "sets in total")
    plotAccCounts(accTable(), title)
  })
  
  # Dendrogram

  # reuse this for 3d plots, too
  output$plotlabelDend = renderText(paste("Distances between models, over data sets of  size", 
                                        input$minrows, "to", input$maxrows, "rows"))
  
  output$plotDend = renderPlot( {
     title = paste("Model groupings:", dataSets()$nsets, "sets in total")
    plotModelDends(dmatrix(), title)
   })
  
  # 3d plots
  
  output$plotDists <- renderWebGL({
    plotModelDists(dmatrix())
  })
  
})
