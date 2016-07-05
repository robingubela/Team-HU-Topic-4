#############################################
#                                           #
#         Quinlan's C5.0 algorithm          #
#                                           #
#############################################

## Some set-up and data preprocessing

# Initial task: Setting working directionary
project_directory <- "C:/Users/gubelaro.hub/Documents"
setwd(project_directory)
getwd()

# Data import 
load("C:/Users/Robin/Dropbox/APA Seminar - Feature Selection/data_2.10_full.rda")

# Preprocessing
df$returnQuantity <- as.factor(df$returnQuantity)
train <- df[!is.na(df$returnQuantity),] 
rm(df)

# Load C50 library
install.packages("C50")
library("C50")

# Perform C5.0 algorithm on data subset, print summary and save importance ranking
treeModel <- C5.0(x = train[,2:52], y = train$returnQuantity, subset=1:500000)
treeModel
summary(treeModel)
x <- C5imp(treeModel, metric = "splits", pct = TRUE)
x
save(x)