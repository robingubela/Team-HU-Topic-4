#####################################
#                                   #
#    Multicollinearity analysis     #
#                                   #
#####################################

# Data version data_2_8_full with 44 variables (included ready2use generated ones)
# Install some packages upfront
install.packages("caret")
library("caret")
install.packages("igraph")
library("igraph")
install.packages("lattice")
library("lattice")
install.packages("corrplot")
library("corrplot")

# Creating a correlation matrix as data object to be leveraged in the analyses
# To do so, look at the data format of all variables and save them in a numeric and categorical space, respectively.
# Because returnQuantitiy is the dependent variable in terms of the DMC task, we do not want to find out correlation between independent variables and returnQuantity, but just among all independent variables
df_num <- c(8,9,10,12,17,21,22,23,24,28,29,30,31,35,36,37,38,39,40,41,42,43,44)
df_cat <- c(2,3,4,5,6,7,11,13,14,15,18,19,20,25,26,27,32,33,34)

cor_num_S <- cor(df[,df_num],use="pairwise.complete.obs",method="spearman") # With this we have all numeric variables in a data object!
cor_num_P <- cor(df[,df_num],use="pairwise.complete.obs",method="pearson")

# Let's start easy
basicoverview <- symnum(cor_num_S)
basicoverview

# Correlation table as output
round(cor(cor_num_S),2)
# Here we see some significant positive and negative correlations among several variables, such as
# voucherAmount and voucher_ratio (1.00)
# cumsumOrder and cumArticleCount (0.99)
# cumArticleCount and totOrder (0.97) and several others.

# Corrplot()
corrplot(cor_num_S, method = "circle")
# This can be done with more beauty.

g1 <- graph_from_adjacency_matrix(cor_num_P,mode="upper",weighted= T, diag= T)
E(g1)$weight <- sqrt(E(g1)$weight^2)
g2 <- delete.edges(g1, which(E(g1)$weight < 0.75))
g2 <- simplify(g2)
plot.igraph(g2,vertex.size=8,vertex.label.cex=1, vertex.color="black",vertex.label.degree=pi/2,vertex.label.dist=0.4,vertex.label.color="midnightblue",edge.width=3,edge.color="lightblue",vertex.label.family="Calibri",vertex.label.font=2)

corrplot(corr = cor_num_S, method="pie",tl.cex=0.3, diag=F,type="lower")

# cor_num_S with "S" for "Spearman" takes all numeric variables into consideration and thus is our data object for subsequent analyses

# findCorrelation()
# Determine highly correlated variables
# Description: This function searches through a correlation matrix and returns a vector of integers corresponding to columns to remove to reduce pair-wise correlations. 
findCorrelation(cor_num_S, cutoff = .50, verbose= TRUE, names= FALSE, exact = TRUE)
findCorrelation(cor_num_S, cutoff = .50, verbose= TRUE, names=TRUE, exact = TRUE)

findCorrelation(cor_num_S, cutoff = .70, verbose= FALSE, names= FALSE, exact = TRUE)
findCorrelation(cor_num_S, cutoff = .70, verbose= FALSE, names= TRUE, exact = TRUE)
findCorrelation(cor_num_S, cutoff = .70, verbose= TRUE, names= TRUE, exact = TRUE)

findCorrelation(cor_num_S, cutoff = .90, verbose= FALSE, names= FALSE, exact = TRUE)
findCorrelation(cor_num_S, cutoff = .90, verbose= TRUE, names= FALSE, exact = TRUE)
findCorrelation(cor_num_S, cutoff = .90, verbose= TRUE, names= TRUE, exact = TRUE)

########## Multicollinearity and variable selection in the light of variable inflation factors ##########

# Q: What is the optimal set of both basic and generated variables? We have to check multicollinearity among existing explanatory variables.
# So called 'ecological' models - taking every variable into the final model prediction into account - lowers CPU, time and overall accuracy. 
# A simple approach to identify collinearity among explanatory variables is the use of variance inflation factors (VIF).
# The way to do it is calculating VIFs for every variable in-scope (those ready-to-use-ones) and removing the one with the highest value in round 1, recalulate, remove the one with the now highest value in round 2, et.
# The best package to use is the VIF package as it allows stepwise selection. It stops when the upfront defined threshold is met. 

install.packages("car") # for VIF functions
library("car")
install.packages("VIF")
library("VIF")
install.packages("usdm")
library("usdm")

v1 <- vifcor(cor_num_S, th=0,9) # identify collinear variables that should be excluded
v1 # From 24 continuous variables 9 are removed from our optimal feature subset due to collinearity reasons.
re1 <- exclude(cor_num_S,v1) # exclude the collinear variables that were identified in the previous step
re1
v2 <- vifstep(cor_num_S, th=10) # identify collinear variables that should be excluded
v2 # From 24 continuous variables 12 are removed from our optimal feature subset due to collinearity reasons.
re2 <- exclude(cor_num_S, v2) # exclude the collinear variables that were identified in the previous step
re2
#v3 <- vifstep(cor_num_S, th=10) # identify collinear variables that should be excluded
#v3
# Here v3 is not necessary because after the second iteration VIFs scores are below 10 and thus good
re3 <- exclude(cor_num_S) # first, vifstep is called
re3

# VIF > 10 indicates the presenece of multicollinearity.
# The solution is to drop all variables that have a VIF score above 10.