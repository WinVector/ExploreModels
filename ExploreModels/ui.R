# AccuracyCounts
options(rgl.useNULL=TRUE)
library(shiny)
library(shinyRGL)
source("shinyFunctions.R")

urlNote = 'See <a target="_blank" href="http://www.win-vector.com/blog/2014/12/the-geometry-of-classifiers/">The Win-Vector blog</a> for details.'


# Define UI for application that draws a histogram
shinyUI(fluidPage(
  
  # Application title
  titlePanel("Explore Model Behavior"),
  
  # Sidebar for number of class outputs : 
  # interval around it; right now set to +/- 10%
  sidebarLayout(
    sidebarPanel(
      numericInput("minrows",
                   "Minimum data set size:",
                   0,  # initial value
                   min = 0,
                   max = MAXROWS,
                   step = 100), 
      
      numericInput("maxrows",
                   "Maximum data set size:",
                   MAXROWS, # initial value
                   min = 0,
                   max = MAXROWS,
                   step = 100),
      
      hr(),
      
      checkboxGroupInput("nclasses", "Number of target classes (check all that apply)",
                         c("Two-Class" = "2",
                           "3-5 classes" = "3:5",
                           "6-10 classes" = "6:10",
                           "more than 10 classes" = "11:MAXN"),
                         selected = c("2", "3:5", "6:10", "11:MAXN")), # all of them
      
      hr(),
      
      checkboxGroupInput("ratio", "Data set shape (variables vs. rows)",  # intervals ( ..]
                         c("Narrow (Less than 1 variable per 100 rows)" = "c(0, 0.01)",
                           "Moderate (1 to 3 variables per 100 rows)" = "c(0.01, 0.03)",
                           "Wide (3 to 10 variables per 100 rows)" = "c(0.03, 0.1)",
                           "Very wide (More than 10 variables per 100 rows" = "c(0.1, Inf)"),
                         selected = c("c(0, 0.01)", "c(0.01, 0.03)", "c(0.03, 0.1)", "c(0.1, Inf)")) # all of them
    ),
    
    # TODO: tabs for Count Accuracy, Model Distance Dendrogram, Model Distance 3D Plot
    mainPanel (
      tabsetPanel (
         tabPanel("Average Accuracy", list(textOutput("plotlabelAA"),HTML(urlNote)), plotOutput("accPlot")),
         tabPanel("Accuracy Counts", list(textOutput("plotlabelAC"),HTML(urlNote)), plotOutput("countPlot")),
         tabPanel("Model Dendrogram", list(textOutput("plotlabelDend"),HTML(urlNote)), plotOutput("plotDend")),
         tabPanel("Model Distances (3d)", list(textOutput("plotlabelDent"),HTML(urlNote)),  webGLOutput("plotDists"))
      )
    )
    
  ) # end sidebarLayout
)
)

