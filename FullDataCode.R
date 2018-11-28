
###########################################################
# Full dataset
###########################################################


###########################################################
# Preamble
###########################################################

rm(list = ls())#clean space

library(class)# for knn
library(xtable)
library(lattice)
library(scatterplot3d)
library(rgl)
library(ggplot2)
library(plot3D)
library(RColorBrewer)
require(randomForest)
library(e1071)
library(glmnet)
library(dplyr)
library(tidyr)
library(keras)
set.seed(1000)
source('solarfunctions.r') # Load the subroutines

###########################################################
# Read in data
###########################################################
a <- readRDS("a.rds") # flare_assoc_structures_2003-2008.txt
b <- readRDS("b.rds") # flare_op_seg_assoc_structures_1996-2002_2009-2011.txt
##############################################################
# Filter detection to those within 45^o from the disc centre
##############################################################

a <- a[abs(a$Hglon_deg) < 45.01, ]
b <- b[abs(b$Hglon_deg) < 45.01, ]


##############################################################
# Get year information for each detection
##############################################################

# data a
Time <- as.character(a$Time)
time <- unlist(strsplit(Time, "_"))


yearmonth <- time[as.logical(abs(is.even(1 : length(time)) - 1))] 
year <- substr(yearmonth, 1, 4)
year <- as.numeric(year)


month <- substr(yearmonth, 5, 6)
month <- as.numeric(month)

# data b

Time <- as.character(b$Time)
time <- unlist(strsplit(Time, "_"))
yearmonth <- time[as.logical(abs(is.even(1 : length(time)) - 1))] 
year2 <- substr(yearmonth, 1, 4)
year2 <- as.numeric(year2)
month2 <- substr(yearmonth, 5, 6)
month2 <- as.numeric(month2)





##############################################################
# Subset data to only relevant features
##############################################################


B <-  apply(cbind(a$Bmax_G,	abs(a$Bmin_G)), 1, max)

usefulFeatures <- cbind(a$Rval_Mx ,   a$WLsg_GpMm,  a$Lsg_Mm,  a$Lnl_Mm,	
                        a$Bflux_Mx,		a$MxGrad_GpMm,		a$MednGrad,	
                B,	a$Area_Mmsq,	a$Bfluximb,	a$HGlon_wdth,	a$HGlat_wdth,	
                a$DBfluxDt_Mx)

colnames(usefulFeatures) <- c("Rval_Mx" ,   "WLsg_GpMm",  "Lsg_Mm", "Lnl_Mm",	
                          "Bflux_Mx",		"MxGrad_GpMm",		"MednGrad",	
                           "B",	"Area_Mmsq",	"Bfluximb",	"HGlon_wdth",	"HGlat_wdth",	
                           "DBfluxDt_Mx")

bB <-  apply(cbind(b$Bmax_G, abs(b$Bmin_G)), 1, max)

busefulFeatures <- cbind(b$Rval_Mx,   b$WLsg_GpMm,  
                  b$Lsg_Mm,	b$Lnl_Mm,	b$Bflux_Mx,		b$MxGrad_GpMm,	
                  b$MednGrad,	bB,	
                  b$Area_Mmsq,	b$Bfluximb,	b$HGlon_wdth,	
                  b$HGlat_wdth,	b$DBfluxDt_Mx)
	
colnames(busefulFeatures) <- colnames(usefulFeatures)

###########################################################
# Transform features
###########################################################

usefulFeatures[,1] <- log(usefulFeatures[,1]+1)
usefulFeatures[,2] <- log(usefulFeatures[,2]+1)
usefulFeatures[,3] <- log(usefulFeatures[,3]+1)
usefulFeatures[,4] <- log(usefulFeatures[,4]+1)
usefulFeatures[,5] <- log(usefulFeatures[,5]+1)

busefulFeatures[,1] <- log(busefulFeatures[,1]+1)
busefulFeatures[,2] <- log(busefulFeatures[,2]+1)
busefulFeatures[,3] <- log(busefulFeatures[,3]+1)
busefulFeatures[,4] <- log(busefulFeatures[,4]+1)
busefulFeatures[,5] <- log(busefulFeatures[,5]+1)

mabs<- min(abs(usefulFeatures[,13]), abs(busefulFeatures[,13]))
usefulFeatures[usefulFeatures[,13]>0,13] <- log(usefulFeatures[usefulFeatures[,13]>0, 13] - mabs)
usefulFeatures[usefulFeatures[,13]<0,13] <- -log(abs(usefulFeatures[usefulFeatures[,13]<0, 13] + mabs)+1)

busefulFeatures[busefulFeatures[,13]>0,13] <- log(busefulFeatures[busefulFeatures[,13]>0, 13] - mabs)
busefulFeatures[busefulFeatures[,13]<0,13] <- -log(abs(busefulFeatures[busefulFeatures[,13]<0, 13] + mabs)+1)



###########################################################
# subset data into training/validation (data)  and testing sets (dataTest)
###########################################################

data <-  rbind(busefulFeatures[year2 <= 2000, ], usefulFeatures)
flare <- c(b$DidFlare[year2 <= 2000], a$DidFlare)

dataTest <- busefulFeatures[year2 >= 2001 & year2 < 2011, ]
flareTest <- b$DidFlare[year2 >= 2001 & year2 < 2011]


colnames(data)<- colnames(usefulFeatures)
colnames(dataTest)<- colnames(usefulFeatures)


###########################################################
# Plot features in training set.
###########################################################
dtf <- data.frame(cbind(data, flare))
dtf$flare<- as.factor(dtf$flare)

# for(i in 1:13){
#   p <- ggplot(dtf, aes(dtf[,i], fill = flare)) +
#   geom_density(alpha = 0.7) + xlab(names(dtf)[i]) +
#   scale_fill_manual( values = c("black","red"))
#   print(p)
# }

##############################
# Scale data                 #
##############################

data <-as.matrix(scale(data))
means <- attr(data,"scaled:center")
sds<- attr(data,"scaled:scale")
dataTest <- as.matrix(scale(dataTest, center=means, scale=sds))

##############################
# to find feature MR score   #
##############################

nFeatures <- dim(data)[2]

library(BKPC)

#find MR scores
margRelv <- marginalRelevance(data, as.factor(flare))

# plot 
matplot(t(margRelv$score), type = "l", xaxt = "n", ylab = "MR score") #xlab = "features",
axis(1, at = 1 : nFeatures, labels = colnames(data),las = 2, cex.axis=0.7)

# table
xtable(t(t(colnames(data)[margRelv$bestVars])))

# rank of the features
featureRank <- margRelv$rank

###########################################################
# random splits of training/validation 
###########################################################
nResamp <- 50 # Number of random splits

n1 <- 200 # Number of flares in the resampled dataset
n0 <- 200 # Number of non-flares in the resampled dataset

N1  = sum(flare == 1) # Number of flares in the training dataset
N0 = sum(flare == 0) # Number of non-flares in the resampled dataset

sprev <- n1/(n1 + n0) # For normalizing betas
prev <- N1/(N1 + N0) # For normalizing betas

indexSet <- subSample(nResamp = nResamp, n1 = n1, n0 = n0, N1  = N1, N0 = N0)


####################################################
#  DNNs
####################################################


flare1 <- to_categorical(flare)[,2]
flareTest1 <- to_categorical(flareTest)[,2]


# Initialize a sequential model
model1 <- keras_model_sequential()
# Add layers to the model
# LR
model1 %>% layer_dense(units = 1, activation = 'sigmoid', input_shape = c(13))
# Print a summary of a model
summary(model1)


# Compile the model
model1 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)


history <- model1 %>% fit(data, flare1,
                          epochs = 200,
                          batch_size = 1024,
                          validation_split = 0.2,
                          class_weight = as.list(c("0" = 1, "1"=20))
)

plot(history)
classes <- model1 %>% predict_classes(dataTest, batch_size = 1024)
table(flareTest1, classes)
prob <- model1 %>% predict_proba(dataTest, batch_size = 1024)
p1Rs <- prob

# 8-4
model2 <- keras_model_sequential()
model2 %>% 
  layer_dense(units = 8, activation = 'tanh', input_shape = c(13)) %>% 
  layer_dense(units = 4, activation = 'tanh') %>% 
  layer_dense(units = 1, activation = 'sigmoid')
# Print a summary of a model
summary(model2)

# Compile the model
model2 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)
history <- model2 %>% fit(data, flare1,
                          epochs = 200,
                          batch_size = 1024,
                          validation_split = 0.2,
                          class_weight = as.list(c("0" = 1, "1"=20))
)

plot(history)
classes <- model2 %>% predict_classes(dataTest, batch_size = 1024)
table(flareTest1, classes)
prob <- model2 %>% predict_proba(dataTest, batch_size = 1024)
p2Rs <- prob

# 16-16

model4 <- keras_model_sequential()

model4 %>% 
  layer_dense(units = 16, activation = 'tanh', input_shape = c(13)) %>% 
  layer_dense(units = 16, activation = 'tanh') %>% 
  layer_dense(units = 1, activation = 'sigmoid')
# Print a summary of a model
summary(model4)


# Compile the model
model4 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

history <- model4 %>% fit(data, flare1,
                          epochs = 200,
                          batch_size = 1024,
                          validation_split = 0.2,
                          class_weight = as.list(c("0" = 1, "1"=20))
)

plot(history)
classes <- model4 %>% predict_classes(dataTest, batch_size = 1024)
table(flareTest1, classes)
prob <- model4 %>% predict_proba(dataTest, batch_size = 1024)
p4Rs <- prob

# 256-32
model6 <- keras_model_sequential()

model6 %>% 
  layer_dense(units = 256, activation = 'tanh', input_shape = c(13)) %>% 
  layer_dense(units = 32, activation = 'tanh') %>% 
  layer_dense(units = 1, activation = 'sigmoid')
# Print a summary of a model
summary(model6)
# Compile the model
model6 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)
history <- model6 %>% fit(data, flare1,
                          epochs = 200,
                          batch_size = 1024,
                          validation_split = 0.2,
                          class_weight = as.list(c("0" = 1, "1"=20))
)

plot(history)
classes <- model6 %>% predict_classes(dataTest, batch_size = 1024)
table(flareTest1, classes)
prob <- model6 %>% predict_proba(dataTest, batch_size = 1024)

p6Rs <- prob
# 31-6-6
model3 <- keras_model_sequential()

model3 %>% 
  layer_dense(units = 13, activation = 'tanh', input_shape = c(13)) %>% 
  layer_dense(units = 6, activation = 'tanh') %>% 
  layer_dense(units = 6, activation = 'tanh') %>% 
  layer_dense(units = 1, activation = 'sigmoid')
# Print a summary of a model
summary(model3)
# Compile the model
model3 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

history <- model3 %>% fit(data, flare1,
                          epochs = 200,
                          batch_size = 1024,
                          validation_split = 0.2,
                          class_weight = as.list(c("0" = 1, "1"=20))
)

plot(history)
classes <- model3 %>% predict_classes(dataTest, batch_size = 1024)
table(flareTest1, classes)
prob <- model3 %>% predict_proba(dataTest, batch_size = 1024)
p3Rs <- prob


rtestp1 <- getSkillScore(p = seq(0.01, 0.9, by = 0.01), pHat = p1Rs, y = flareTest1)
rtestpDNN_8_4 <- getSkillScore(p = seq(0.01, 0.9, by = 0.01), pHat = p2Rs, y = flareTest1)
rtestpDNN_13_6_6 <- getSkillScore(p = seq(0.01, 0.9, by = 0.01), pHat = p3Rs, y = flareTest1)
rtestpDNN_16_16 <- getSkillScore(p = seq(0.01, 0.9, by = 0.01), pHat = p4Rs, y = flareTest1)
rtestpDNN_256_32 <- getSkillScore(p = seq(0.01, 0.9, by = 0.01), pHat = p6Rs, y = flareTest1)


getAUC(rtestp1$TPR, rtestp1$FPR)
round(ltxtable(rtestpDNN_256_32),2)
getAUC(rtestp1$TPR, rtestp1$FPR)
round(ltxtable(rtestpDNN_8_4),2)

max(rtestp1$TSS)
max(rtestpDNN_8_4$TSS)
max(rtestpDNN_16_16$TSS)
max(rtestpDNN_256_32$TSS)
max(rtestpDNN_13_6_6$TSS)


###########################################################
# Datasets for random splits of training/validation 
###########################################################

dtf <- data.frame(cbind(data, flare)) 
dtf$flare<- as.factor(dtf$flare)

#sort data for random sampling
dataSort <- rbind(data[flare == 0, ], data[flare == 1, ])
flareSort <- sort(flare)

dtfSort <- data.frame(dataSort)
dtfSort$flare<- as.factor(sort(flare))


##################################
#  classifier                    #
##################################

# initialise

nFeat <- 4  # no of features to include + 1 (intercept)

betas <- matrix(0, nResamp, nFeat)
# pValid <- matrix(0, (dim(data)[1] - n0 - n1) , nResamp)
pTest <- matrix(0, dim(dataTest)[1] , nResamp)

# pValidSVM <- matrix(0, (dim(data)[1] - n0 - n1) , nResamp)
pTestSVM <- matrix(0, dim(dataTest)[1] , nResamp)

#######################
# train the classifier
#######################


for (i in 1 : nResamp) 
{
  index <- indexSet[i, ]
  solar_glm_0 <- glm(as.factor(flareSort[index]) ~ dataSort[index, featureRank < nFeat],  family = binomial())
  betas[i, ] <- as.matrix(coef(solar_glm_0))
}


# normalise intercept
betas[,1] <- betas[,1] + log(prev/(1-prev)) - log(sprev/(1-sprev))

for (i in 1 : nResamp) 
{
  pTemp <- cbind(rep(1, dim(dataTest)[1]), dataTest[ , featureRank < nFeat]) %*% betas[i, ] 
  pTest[, i] <- logitInv(pTemp)
}


for (i in 1 : nResamp) 
{
  index <- indexSet[i, ]
  astro.svm <- svm(flare ~ . , data = dtfSort[index,], kernel = "radial",gamma = 0.03, probability = TRUE)
  
  pTemp <- predict(astro.svm, dataTest, probability = TRUE) 
  pTemp <- attributes(pTemp)$probabilities
  pTestSVM[, i] <- matrix(pTemp[,2], length(pTemp[,2]), 1)
}



#####################################
# resample results testing set
#####################################

rtestLR <- getSkillScore(p= seq(0.01, 0.9, by = 0.01), pHat = pTest, y = flareTest)
rtestSVM <- getSkillScore(p= seq(0.01, 0.9, by = 0.01), pHat = pTestSVM, y = flareTest)


########################################################
# glm using entire training set - no resampling 
########################################################


nFeat <- 4

solar_glm_0 <- glm(as.factor(flare) ~ data[ , featureRank < nFeat],  family = binomial())
summary(solar_glm_0)

solar_glm_2 <- glm(as.factor(flare) ~ data,  family = binomial())
summary(solar_glm_2)



pTempB <- cbind(rep(1, dim(dataTest)[1]), dataTest[ ,featureRank < nFeat]) %*% coef(solar_glm_0)# pTempB <- cbind(rep(1, dim(dataTest)[1]), dataTest[ ,topfeats==1 ]) %*% coef(solar_glm_0)
pTempB2 <- cbind(rep(1, dim(dataTest)[1]),dataTest) %*% coef(solar_glm_2)

pTestF <- logitInv(pTempB)
pTest2 <- logitInv(pTempB2)
rtestLR3 <- getSkillScore(p= seq(0.01, 0.9, by = 0.01), pHat = pTestF, y = flareTest)
rtestLR13 <- getSkillScore(p= seq(0.01, 0.9, by = 0.01), pHat = pTest2, y = flareTest)
max(rtestLR$TSS)
max(rtestLR3$TSS)
max(rtestLR13$TSS)
max(rtestSVM$TSS)
###########################################################
# LASSO
###########################################################
# xf <- model.matrix(as.factor(flare) ~ data)
# yf <- as.factor(flare)
# 
# 
# grid<- c(0.03, 0.04, 0.04,  0.12)
# i <-1
# 
# lasso.fit <- glmnet(xf,yf,alpha=1,
#                     lambda = grid[i], family="binomial")
# 
# betal <- as.matrix(coef(lasso.fit))
# betal
# betal <- betal[-2]
# pTempl <- cbind(rep(1, dim(dataTest)[1]),dataTest) %*% betal
# pTestL <- logitInv(pTempl)
# rtestL <- getSkillScore(p= seq(0.01, 0.9, by = 0.01), pHat = pTestL, y = flareTest)
# 
# 
# 
#####################################
# For RF ---------------------------
#####################################


astro.rf <- randomForest(flare ~ . , data = dtf,  type = classification, ntree = 500, importance=TRUE)


pred <- predict(astro.rf,dataTest, type = "prob") 
pRF <- matrix(pred[,2], length(pred[,2]),1)

vi <- importance(astro.rf, type = 2)
vi <- cbind(vi, t(margRelv$score))


rtestRF <- getSkillScore(p= seq(0.01, 0.9, by = 0.01), pHat = pRF, y = flareTest)
max(rtestRF$TSS)


#####################################
# COMBINE ALL RESULTS
#####################################


p <- seq(0.01, 0.9, by = 0.01)

##################################################################################
# SMALL TRAINING SETS
##################################################################################


  
table2 <- round(ltxtable(rtestLR),2)
table3 <- cbind(table2[, 1:2], paste("(", table2[,3],"," ,table2[, 4], ")"), table2[, c(5,8,11,14)],paste("(", table2[,15],"," ,table2[, 16], ")"))
print(xtable(table3), include.rownames = F)

AUCtest <- getAUC(rtestLR$TPR, rtestLR$FPR)
AUCtestSVM <- getAUC(rtestSVM$TPR, rtestSVM$FPR)

AUCtestmed <- median(AUCtest)
AUCtestL <- quantile(AUCtest, prob = 0.025)
AUCtestU <- quantile(AUCtest, prob = 0.975)


AUCtestmed <- median(AUCtestSVM)
AUCtestL <- quantile(AUCtestSVM, prob = 0.025)
AUCtestU <- quantile(AUCtestSVM, prob = 0.975)



##################################################################################
# FULL TRAINING SET
##################################################################################

AUCtestfull <- getAUC(rtestLR3$TPR, rtestLR3$FPR)
AUCtestfullRF <- getAUC(rtestRF$TPR, rtestRF$FPR)

table2 <- round(ltxtable(rtestLR),2)
table3 <- cbind(table2[, 1:2],  table2[, c(5,8,11,14)])
print(xtable(table3[c(c(1:9), seq(10, 90, by = 10)), ]), include.rownames = F)

# LASSO
# max(as.numeric(rtestL$TSS))
# getAUC(rtestL$TPR, rtestL$FPR)


# PLOTS

p <- seq(0.01, 0.9, by = 0.01)
LR <- apply(rtestLR$TSS, 2, median)
SVM <- apply(rtestSVM$TSS, 2, median)
LR3 <- as.numeric(rtestLR3$TSS)
LR13 <- as.numeric(rtestLR13$TSS)
RF <- as.numeric(rtestRF$TSS)
DNN_8_4 <- as.numeric(rtestpDNN_8_4$TSS)
DNN_13_6_6 <- as.numeric(rtestpDNN_13_6_6$TSS)
DNN_16_16 <- as.numeric(rtestpDNN_16_16$TSS)
DNN_256_32 <- as.numeric(rtestpDNN_256_32$TSS)
xL <- apply(rtestLR$TSS, 2, quantile, prob = 0.025)
xU <- apply(rtestLR$TSS, 2,  quantile, prob = 0.975)
dfplot <- data.frame(cbind(LR, SVM,LR3,  LR13,RF,DNN_8_4,DNN_13_6_6 ,DNN_16_16, DNN_256_32, xL, xU, p))

dfplot <-  dfplot %>%
  gather(Classifier, TSS,  c(LR, SVM,LR3, LR13, RF, DNN_8_4,DNN_13_6_6,DNN_256_32 ,DNN_16_16))

dfplot$Classifier <- factor(dfplot$Classifier, levels = levels(as.factor(dfplot$Classifier))[c(5,7, 6,8,9,4,2,3,1)])


ggplot(dfplot, aes(x = p, y = TSS, col = Classifier, lty = Classifier)) + 
  geom_line()+xlim(0, 1)+ylim(0, 0.85)+ ylab("TSS") +ggtitle("TSS: full SMART dataset")+ 
  geom_ribbon(aes(x = p, ymin=xU, ymax=xL), alpha=0.2, inherit.aes = FALSE)
  


# ggsave("TSScompareFullV3.pdf", width = 5, height = 3)


##################################################################################


getAUC(rtestLR3$TPR, rtestLR3$FPR)
getAUC(rtestLR13$TPR, rtestLR13$FPR)
getAUC(rtestRF$TPR, rtestRF$FPR)
getAUC(rtestpDNN_8_4$TPR, rtestpDNN_8_4$FPR)
getAUC(rtestpDNN_16_16$TPR, rtestpDNN_16_16$FPR)
getAUC(rtestpDNN_256_32$TPR, rtestpDNN_256_32$FPR)
getAUC(rtestpDNN_13_6_6$TPR, rtestpDNN_13_6_6$FPR)

LR <- c(0,apply(rtestLR$FPR, 2, median),1)
SVM <- c(0,apply(rtestSVM$FPR, 2, median),1)
LR3 <- c(0,rtestLR3$FPR,1)
LR13 <- c(0,rtestLR13$FPR,1)
RF <- c(0,rtestRF$FPR,1)
DNN_8_4 <- c(0,rtestpDNN_8_4$FPR,1)
DNN_13_6_6 <- c(0,rtestpDNN_13_6_6$FPR,1)
DNN_16_16 <- c(0,rtestpDNN_16_16$FPR,1)
DNN_256_32 <- c(0,rtestpDNN_256_32$FPR,1)
dfplotx <- data.frame(cbind(LR, SVM,LR3,  LR13,RF,DNN_8_4,DNN_13_6_6 ,DNN_16_16,DNN_256_32)) #, DNN_256_32
dfplotx <-  dfplotx %>%
  gather(Classifier, FPR,  c(LR, SVM,LR3,  LR13,RF,DNN_8_4,DNN_13_6_6 ,DNN_16_16, DNN_256_32))

LR <- c(0,apply(rtestLR$TPR, 2, median),1)
SVM <- c(0,apply(rtestSVM$TPR, 2, median),1)
LR3 <- c(0,rtestLR3$TPR,1)
LR13 <- c(0,rtestLR13$TPR,1)
RF <- c(0,rtestRF$TPR,1)
DNN_8_4 <- c(0,rtestpDNN_8_4$TPR,1)
DNN_13_6_6 <- c(0,rtestpDNN_13_6_6$TPR,1)
DNN_16_16 <- c(0,rtestpDNN_16_16$TPR,1)
DNN_256_32 <- c(0,rtestpDNN_256_32$TPR,1)
dfploty <- data.frame(cbind(LR, SVM,LR3,  LR13,RF,DNN_8_4,DNN_13_6_6 ,DNN_16_16, DNN_256_32))

dfploty <-  dfploty %>%
  gather(Classifier, TPR,  c(LR, SVM,LR3,  LR13,RF,DNN_8_4,DNN_13_6_6 ,DNN_16_16, DNN_256_32))
dfplot <- data.frame(cbind(dfplotx, TPR = dfploty$TPR))

dfplot$Classifier <- factor(dfplot$Classifier, levels = levels(as.factor(dfplot$Classifier))[c(5,7, 6,8,9,4,2,3,1)])#c(4,6, 5,7,8,3,2,1)c(5,7, 6,8,9,4,2,3,1)

ggplot(dfplot, aes(x = FPR, y = TPR, col = Classifier, lty = Classifier)) + geom_line()+
  xlim(0, 1)+ylim(0, 1)+ ylab("TPR")+ xlab("FPR")+ggtitle("ROC: full SMART dataset")
# ggsave("ROCcompareV3.pdf", width = 5, height = 3)




youden <- rtestLR3$TPR +rtestLR3$TNR  - 1 # J = sensitivity + specificity − 1 = TSS
plot(seq(0.01, 0.9, by = 0.01), rtestLR3$TSS , col = 0)
lines(seq(0.01, 0.9, by = 0.01), rtestLR3$TSS,  col = 1, lty = 1, lwd = 2)
lines(seq(0.01, 0.9, by = 0.01), youden,  col = 3, lty = 1, lwd = 2)


dfplot <- data.frame(p = seq(0.01, 0.9, by = 0.01), ACC = t(rtestLR3$ACC), TPR = t(rtestLR3$TPR), TNR =  t(rtestLR3$TNR), TSS = t(rtestLR3$TSS), HSS = t(rtestLR3$HSS))


dfplot <-  dfplot %>%
  gather(PerformanceMeasure, Score,  ACC:HSS)
ggplot(dfplot, aes(x = p, y = Score, col = PerformanceMeasure, lty = PerformanceMeasure)) + geom_line()+
  geom_vline(xintercept = prev, col = "grey")+ xlim(0, 1)+ylim(0, 1)+ ylab("")+ xlab("p")+ggtitle("Performance Measures")+
  theme(legend.title=element_blank())
# ggsave("ChoosePV3.pdf", width = 5, height = 3)

##################################################################################

LR <- apply(rtestLR$HSS, 2, median)
SVM <- apply(rtestSVM$HSS, 2, median)
LR3 <- as.numeric(rtestLR3$HSS)
LR13 <- as.numeric(rtestLR13$HSS)
RF <- as.numeric(rtestRF$HSS)
DNN_8_4 <- as.numeric(rtestpDNN_8_4$HSS)
DNN_13_6_6 <- as.numeric(rtestpDNN_13_6_6$HSS)
DNN_16_16 <- as.numeric(rtestpDNN_16_16$HSS)
DNN_256_32 <- as.numeric(rtestpDNN_256_32$HSS)

dfmax <- data.frame(cbind(LR, LR3,  LR13,RF,SVM,DNN_8_4,DNN_13_6_6 ,DNN_16_16,DNN_256_32, p))# 

xtable(t(t(apply(dfmax[,1:9], 2, max))),digits = 2)#1:9

maxp <- maxTSS
for(i in 1:9)maxp[i]<-dfmax[dfmax[,i]==maxTSS[i],10]

xtable(t(t(apply(dfmax[,1:9], 2, max))),digits = 2)


apply(rtestLR$TSS, 2, quantile, prob = 0.5)==max(apply(rtestLR$TSS, 2, quantile, prob = 0.5))

apply(rtestLR$TSS, 2, quantile, prob = 0.025)[5]
apply(rtestLR$TSS, 2, quantile, prob = 0.5)[5]
apply(rtestLR$TSS, 2, quantile, prob = 0.975)[5]


apply(rtestSVM$TSS, 2, quantile, prob = 0.5)==max(apply(rtestSVM$TSS, 2, quantile, prob = 0.5))

apply(rtestSVM$TSS, 2, quantile, prob = 0.025)[61]
apply(rtestSVM$TSS, 2, quantile, prob = 0.5)[61]
apply(rtestSVM$TSS, 2, quantile, prob = 0.975)[61]



apply(rtestLR$HSS, 2, quantile, prob = 0.5)==max(apply(rtestLR$HSS, 2, quantile, prob = 0.5))

apply(rtestLR$HSS, 2, quantile, prob = 0.025)[29]
apply(rtestLR$HSS, 2, quantile, prob = 0.5)[29]
apply(rtestLR$HSS, 2, quantile, prob = 0.975)[29]


apply(rtestSVM$HSS, 2, quantile, prob = 0.5)==max(apply(rtestSVM$HSS, 2, quantile, prob = 0.5))

apply(rtestSVM$HSS, 2, quantile, prob = 0.025)[86]
apply(rtestSVM$HSS, 2, quantile, prob = 0.5)[86]
apply(rtestSVM$HSS, 2, quantile, prob = 0.975)[86]
# save.image("SMART1.RData") 



####################################################################
#  plots: Data skip the log transformation                         #
####################################################################


i <- nResamp
  index <- indexSet[i, ]
  dataTrain <- dataSort[index, ]
  wtr <- flareSort[index]
 
scatter3D(dataTrain[ ,featureRank < 4][,1],dataTrain[ ,featureRank < 4][,2], dataTrain[ ,featureRank < 4][, 3],
          colvar = as.integer(wtr),col = c("#2E2E2E", "#CD3333"), theta = 20, phi = 10, bty = "g",
          pch = 1, cex = 0.6, colkey = FALSE,  xlab = colnames(dataTrain[ , featureRank < 4])[1],
          ylab =colnames(dataTrain[ , featureRank < 4])[2], zlab = colnames(dataTrain[ , featureRank < 4])[3])
# dev.print(png, "3dTrain.png", res=600, height=8, width=8,  units="cm")

scatter3D(dataTest[ ,featureRank < 4][,1],dataTest[ ,featureRank < 4][,2], dataTest[ ,featureRank < 4][, 3],
          colvar = as.integer(flareTest),col = c("#2E2E2E", "#CD3333"), theta = 20, phi = 10, bty = "g",
          pch = 1, cex = 0.6, colkey = FALSE, xlim = c(0, 500),  xlab = colnames(dataTrain[ , featureRank < 4])[1],
          ylab =colnames(dataTrain[ , featureRank < 4])[2], zlab = colnames(dataTrain[ , featureRank < 4])[3])
# dev.print(png, "3dTest.png", res=600, height=8, width=8,  units="cm")

dtf <- data.frame(cbind(data, flare))
dtf$flare<- as.factor(dtf$flare)
# names(dtf)


plot(dtf$MxGrad_GpMm, dtf$Rval_Mx, col = dtf$flare)
ggplot(dtf, aes(MxGrad_GpMm, fill = flare)) +
  geom_density(alpha = 0.7) +
  xlim(0, 1500)+
  scale_fill_manual( values = c("black","red"))
# ggsave("MxGradGp.pdf", width = 4, height = 3)

ggplot(dtf, aes(log(Rval_Mx+1), fill = flare)) +
  geom_density(alpha = 0.7) +
  # xlim(0, 40)+
  scale_fill_manual( values = c("black","red"))
# ggsave("Rval_Mx.pdf", width = 4, height = 3)

ggplot(dtf, aes(log(Bflux_Mx), fill = flare)) +
  geom_density(alpha = 0.7) +
  xlim(40, 60)+
  scale_fill_manual( values = c("black","red"))
# ggsave("Bflux_Mx.pdf", width = 4, height = 3)


