#########################################################
#                                                       #
#           Feature selection: varSelRF package         #
#                                                       #
#########################################################

# Package description: Variable selection from random forests using both backwards variable elimination (for the selection
#  of small sets of non-redundant variables) and selection based on the importance spectrum (somewhat similar to scree plots;
#  for the selection of large, potentially highly-correlated variables). Main applications in high-dimensional data (e.g.,
#  microarray data, and other genomics and proteomics applications).

## Some set-up and data preprocessing

# Initial task: Setting working directionary
project_directory <- "C:/Users/gubelaro.hub/Documents"
setwd(project_directory)
getwd()

# Data import 
load("C:/Users/Robin/Dropbox/APA Seminar - Feature Selection/data_2.10_full.rda")

# Move reponse variable "returnQuantity" to the front of the data set, to enhance clear understanding for prediction
#col_idx <- grep("returnQuantity", names(df), )
#df <- df[, c(col_idx, (1:ncol(df))[-col_idx])]
#names(df)

# Preprocessing
df$returnQuantity <- as.factor(df$returnQuantity)
train1 <- df[!is.na(df$returnQuantity),] 
rm(df)

# Install/Load the package
install.packages("varSelRF")
library("varSelRF")

# Variable selection with Random Forest
rf.vs1 <- varSelRF(xdata = train1, Class = train$returnQuantity, ntree = 30, mtryFactor = 3, ntreeIterat = 5, vars.drop.frac = 0.2)
rf.vs1
plot(rf.vs1)
plot(rf.vs1, nvar = 30, which = c(1, 2))

#randomVarImpsRF
#Variable importances from random forest on permuted class labels. Return variable importances from random forests fitted to data sets like the original except class labels have been randomly permuted
randomVarImpsRF(train1, train$returnQuantity, rf.vs1, numrandom = 100,
                whichImp = "impsUnscaled", usingCluster = TRUE,
                TheCluster = NULL)