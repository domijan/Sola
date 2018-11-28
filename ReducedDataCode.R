

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

############################################################
# Read in data
###########################################################

a <- read.table("smart_19960421_20110110_noaa_required.txt", header = TRUE)


##############################################################
# Filter detection to those within 45^o from the disc centre
##############################################################
a <- a[abs(a$Hglon_deg) < 45.01, ]


##############################################################
# Get year information for each detection
##############################################################

Time <- as.character(a$Time)
time <- unlist(strsplit(Time, "_"))


yearmonth <- time[as.logical(abs(is.even(1 : length(time)) - 1))] 
year <- substr(yearmonth, 1, 4)
month <- substr(yearmonth, 5, 6)
year <- as.numeric(year)
month <- as.numeric(month)



##############################################################
# Subset data to only relevant features
##############################################################

attach(a)

B <-  apply(cbind(Bmax_G,	abs(Bmin_G)), 1, max)

usefulFeatures <- cbind(Rval_Mx,  WLsg_GpMm,  Lsg_Mm,  Lnl_Mm,	
                Bflux_Mx,		MxGrad_GpMm,	
                MednGrad,	B,	Area_Mmsq,	Bfluximb,	
                HGlon_wdth,	HGlat_wdth,	 DBfluxDt_Mx, NOAA_assigned)

###########################################################
# Transform features
###########################################################

usefulFeatures <- usefulFeatures[, 1:13]
usefulFeatures[,1] <- log(usefulFeatures[,1]+1)
usefulFeatures[,2] <- log(usefulFeatures[,2]+1)
usefulFeatures[,3] <- log(usefulFeatures[,3]+1)
usefulFeatures[,4] <- log(usefulFeatures[,4]+1)
usefulFeatures[,5] <- log(usefulFeatures[,5]+1)

mabs<- min(abs(usefulFeatures[,13]))
usefulFeatures[usefulFeatures[,13]>0,13] <- log(usefulFeatures[usefulFeatures[,13]>0, 13] - mabs)
usefulFeatures[usefulFeatures[,13]<0,13] <- -log(abs(usefulFeatures[usefulFeatures[,13]<0, 13] + mabs)+1)
summary( usefulFeatures[,13])


###########################################################################
# subset data into training/validation (data)  and testing sets (dataTest)
###########################################################################


data <- usefulFeatures[year <= 2000 |(year >= 2003 & year < 2009), ]
flare <- a$DidFlare[year <= 2000 |(year >= 2003 & year < 2009)]

dataTest <- usefulFeatures[(year >= 2001 & year < 2003)| (year >= 2009 & year < 2011), ]
flareTest <- a$DidFlare[(year >= 2001 & year < 2003)| (year >= 2009 & year < 2011)]


###########################################################
# Plot features in training set.
###########################################################
dtf <- data.frame(cbind(data, flare))
dtf$flare<- as.factor(dtf$flare)


# for(i in 1:13){
#   p <- ggplot(dtf, aes(dtf[,i], fill = flare)) +
#     geom_density(alpha = 0.7) + xlab(names(dtf)[i]) +
#     scale_fill_manual( values = c("black","red"))
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
matplot(t(margRelv$score), type = "l", xaxt = "n",  ylab = "MR score") #xlab = "features",
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
                          class_weight = as.list(c("0" = 1, "1"=5))
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
                          class_weight = as.list(c("0" = 1, "1"=5))
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
                          class_weight = as.list(c("0" = 1, "1"=5))
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
                          class_weight = as.list(c("0" = 1, "1"=5))
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
                          class_weight = as.list(c("0" = 1, "1"=5))
)

plot(history)
# classes <- model3 %>% predict_classes(dataTest, batch_size = 1024)
# table(flareTest1, classes)
prob <- model3 %>% predict_proba(dataTest, batch_size = 1024)
p3Rs <- prob



rtestp1 <- getSkillScore(p = seq(0.01, 0.9, by = 0.01), pHat = p1Rs, y = flareTest1)
rtestDNN_8_4 <- getSkillScore(p = seq(0.01, 0.9, by = 0.01), pHat = p2Rs, y = flareTest1)
rtestDNN_13_6_6 <- getSkillScore(p = seq(0.01, 0.9, by = 0.01), pHat = p3Rs, y = flareTest1)
rtestDNN_16_16 <- getSkillScore(p = seq(0.01, 0.9, by = 0.01), pHat = p4Rs, y = flareTest1)
rtestDNN_256_32 <- getSkillScore(p = seq(0.01, 0.9, by = 0.01), pHat = p6Rs, y = flareTest1)


getAUC(rtestp1$TPR, rtestp1$FPR)
round(ltxtable(rtestDNN_256_32),2)
getAUC(rtestp1$TPR, rtestp1$FPR)
round(ltxtable(rtestDNN_8_4),2)

max(rtestp1$TSS)
max(rtestDNN_8_4$TSS)
max(rtestDNN_16_16$TSS)
max(rtestDNN_256_32$TSS)
max(rtestDNN_13_6_6$TSS)


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

pTest <- matrix(0, dim(dataTest)[1] , nResamp)

pTestSVM <- matrix(0, dim(dataTest)[1] , nResamp)


#######################
# train the LR classifier
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
  astro.svm <- svm(flare ~ . , data = dtfSort[index,], kernel = "radial",gamma = 0.01, probability = TRUE)
  
  pTemp <- predict(astro.svm, dataTest, probability = TRUE)
  pTemp <- attributes(pTemp)$probabilities
  pTestSVM[, i] <- matrix(pTemp[,2], length(pTemp[,2]), 1)
}




###########################################################
# SVM: visualise kernels
###########################################################
# 
# index <- indexSet[i, ]
# dataTrain <- dataSort[index, ]
# wtr <- as.factor(flareSort[index])
# 
# library(kernlab)
# 
# kfunc <- laplacedot(sigma = 0.01)
# Ktrain <- kernelMatrix(kfunc, dataTrain)
# image(Ktrain)
# 
# kfunc <- anovadot(sigma = 0.01, degree = 1)
# Ktrain <- kernelMatrix(kfunc, dataTrain)
# image(Ktrain)
# 
# kfunc <- rbfdot(sigma = 0.01)
# Ktrain <- kernelMatrix(kfunc, dataTrain)
# image(Ktrain)
# 
# 

#####################################
# results validation set
#####################################

#####################################
# resample results testing set
#####################################
rtestLR <- getSkillScore(p= seq(0.01, 0.9, by = 0.01), pHat = pTest, y = flareTest)
max(rtestLR$TSS)
rtestSVM <- getSkillScore(p= seq(0.01, 0.9, by = 0.01), pHat = pTestSVM, y = flareTest)
max(rtestSVM$TSS)




########################################################
# glm using entire training set - no resampling 
########################################################


solar_glm_0 <- glm(flare ~ ., data = dtf[ , c(featureRank < nFeat, TRUE)],  family = binomial())
summary(solar_glm_0)

solar_glm_2 <- glm(flare ~ ., data = dtf,  family = binomial())
summary(solar_glm_2)

pTempB <- cbind(rep(1, dim(dataTest)[1]), dataTest[ ,featureRank < nFeat]) %*% coef(solar_glm_0)
pTempB2 <- cbind(rep(1, dim(dataTest)[1]),dataTest) %*% coef(solar_glm_2)

pTestLR3 <- logitInv(pTempB)
pTestLR13 <- logitInv(pTempB2)

rtestLR3 <- getSkillScore(p= seq(0.01, 0.9, by = 0.01), pHat = pTestLR3, y = flareTest)
rtestLR13 <- getSkillScore(p= seq(0.01, 0.9, by = 0.01), pHat = pTestLR13, y = flareTest)

max(rtestLR3$TSS)
max(rtestLR13$TSS)


###########################################################
# LASSO
###########################################################
xf <- model.matrix(as.factor(flare) ~ data)
yf <- as.factor(flare)


grid<- c( 0.01, 0.02, 0.08, 0.1,  0.16, 0.17, 0.175)
i <- 1
lasso.fit <- glmnet(xf,yf,alpha=1,
                    lambda = grid[i], family="binomial")

betal <- as.matrix(coef(lasso.fit))
betal
betal <- betal[-2] # int incl twice

pTempl <- cbind(rep(1, dim(dataTest)[1]),dataTest) %*% betal
pTestL <- logitInv(pTempl)
rtestL <- getSkillScore(p= seq(0.01, 0.9, by = 0.01), pHat = pTestL, y = flareTest)
max(rtestL$TSS)


#####################################
# For RF ---------------------------
#####################################



astro.rf <- randomForest(flare ~ . , data = dtf,  type = classification, ntree = 500, importance=TRUE)


pred <- predict(astro.rf,dataTest, type = "prob") 
pRF <- matrix(pred[,2], length(pred[,2]),1)

rtestRF <- getSkillScore(p= seq(0.01, 0.9, by = 0.01), pHat = pRF, y = flareTest)
max(rtestRF$TSS)
importance(astro.rf, type = 2)[order(importance(astro.rf, type = 2)),]                                                                 
                                                                      
                                                                      
#####################################
# COMBINE ALL RESULTS
#####################################


AUCtest <- getAUC(rtestLR$TPR, rtestLR$FPR)
AUCtestSVM <- getAUC(rtestSVM$TPR, rtestSVM$FPR)

AUCtestmed <- median(AUCtest)
AUCtestL <- quantile(AUCtest, prob = 0.025)
AUCtestU <- quantile(AUCtest, prob = 0.975)
AUCtestmed
AUCtestL
AUCtestU

AUCtestmed <- median(AUCtestSVM)
AUCtestL <- quantile(AUCtestSVM, prob = 0.025)
AUCtestU <- quantile(AUCtestSVM, prob = 0.975)
AUCtestmed
AUCtestL
AUCtestU



##################################################################################
# FULL TRAINING SET
##################################################################################

AUCtestfull <- getAUC(rtestLR3$TPR, rtestLR3$FPR)
AUCtestfullRF <- getAUC(rtestRF$TPR, rtestRF$FPR)

table2 <- round(ltxtable(rtestLR13),2)
table3 <- cbind(table2[, 1:2],  table2[, c(5,8,11,14)])
print(xtable(table3[c(c(1:9), seq(10, 90, by = 10)), ]), include.rownames = F)

table2 <- round(ltxtable(rtestLR),2)
table3 <- cbind(table2[, 1:2], paste("(", table2[,3],"," ,table2[, 4], ")"), table2[, c(5,8,11,14)],paste("(", table2[,15],"," ,table2[, 16], ")"))
print(xtable(table3[c(c(1:9), seq(10, 90, by = 10)), ]), include.rownames = F)


##################################################################################
# PLOTS
##################################################################################


p <- seq(0.01, 0.9, by = 0.01)
LR <- apply(rtestLR$TSS, 2, median)
SVM <- apply(rtestSVM$TSS, 2, median)
LR3 <- as.numeric(rtestLR3$TSS)
LR13 <- as.numeric(rtestLR13$TSS)
RF <- as.numeric(rtestRF$TSS)
DNN_8_4 <- as.numeric(rtestDNN_8_4$TSS)
DNN_13_6_6 <- as.numeric(rtestDNN_13_6_6$TSS)
DNN_16_16 <- as.numeric(rtestDNN_16_16$TSS)
DNN_256_32 <- as.numeric(rtestDNN_256_32$TSS)
xL <- apply(rtestLR$TSS, 2, quantile, prob = 0.025)
xU <- apply(rtestLR$TSS, 2,  quantile, prob = 0.975)
dfplot <- data.frame(cbind(LR, SVM,LR3,  LR13,RF,DNN_8_4,DNN_13_6_6 ,DNN_16_16, DNN_256_32, xL, xU, p))

dfplot <-  dfplot %>%
  gather(Classifier, TSS,  c(LR, SVM,LR3, LR13, RF, DNN_8_4,DNN_13_6_6 ,DNN_16_16, DNN_256_32))

dfplot$Classifier <- factor(dfplot$Classifier, levels = levels(as.factor(dfplot$Classifier))[c(5,7,6,8,9,4,2,3,1)])


ggplot(dfplot, aes(x = p, y = TSS, col = Classifier, lty = Classifier)) + 
  geom_line(lwd = .6)+xlim(0, 1)+ylim(0, 0.85)+ ylab("TSS") +ggtitle("TSS: NOAA ARs")+ 
  geom_ribbon(aes(x = p, ymin=xU, ymax=xL), alpha=0.2, inherit.aes = FALSE)



# ggsave("TSScompareRDV3.pdf", width = 5, height = 3)

##################################################################################

LR <- c(0,apply(rtestLR$FPR, 2, median),1)
SVM <- c(0,apply(rtestSVM$FPR, 2, median),1)
LR3 <- c(0,rtestLR3$FPR,1)
LR13 <- c(0,rtestLR13$FPR,1)
RF <- c(0,rtestRF$FPR,1)
DNN_8_4 <- c(0,rtestDNN_8_4$FPR,1)
DNN_13_6_6 <- c(0,rtestDNN_13_6_6$FPR,1)
DNN_16_16 <- c(0,rtestDNN_16_16$FPR,1)
DNN_256_32 <- c(0,rtestDNN_256_32$FPR,1)
dfplotx <- data.frame(cbind(LR, SVM,LR3,  LR13,RF,DNN_8_4,DNN_13_6_6 ,DNN_16_16, DNN_256_32))
dfplotx <-  dfplotx %>%
  gather(Classifier, FPR,  c(LR, SVM,LR3,  LR13,RF,DNN_8_4,DNN_13_6_6 ,DNN_16_16, DNN_256_32))

LR <- c(0,apply(rtestLR$TPR, 2, median),1)
SVM <- c(0,apply(rtestSVM$TPR, 2, median),1)
LR3 <- c(0,rtestLR3$TPR,1)
LR13 <- c(0,rtestLR13$TPR,1)
RF <- c(0,rtestRF$TPR,1)
DNN_8_4 <- c(0,rtestDNN_8_4$TPR,1)
DNN_13_6_6 <- c(0,rtestDNN_13_6_6$TPR,1)
DNN_16_16 <- c(0,rtestDNN_16_16$TPR,1)
DNN_256_32 <- c(0,rtestDNN_256_32$TPR,1)
dfploty <- data.frame(cbind(LR, SVM,LR3,  LR13,RF,DNN_8_4,DNN_13_6_6 ,DNN_16_16, DNN_256_32))

dfploty <-  dfploty %>%
  gather(Classifier, TPR,  c(LR, SVM,LR3,  LR13,RF,DNN_8_4,DNN_13_6_6 ,DNN_16_16, DNN_256_32))
dfplot <- data.frame(cbind(dfplotx, TPR = dfploty$TPR))

dfplot$Classifier <- factor(dfplot$Classifier, levels = levels(as.factor(dfplot$Classifier))[c(5,7,6,8,9,4,2,3,1)])

ggplot(dfplot, aes(x = FPR, y = TPR, col = Classifier, lty = Classifier)) + geom_line()+
  xlim(0, 1)+ylim(0, 1)+ ylab("TPR")+ xlab("FPR")+ggtitle("ROC: NOAA ARs")
# ggsave("ROCcompareRDV3.pdf", width = 5, height = 3)




dfplot <- data.frame(p = seq(0.01, 0.9, by = 0.01), ACC = t(rtestLR3$ACC), TPR = t(rtestLR3$TPR), TNR =  t(rtestLR3$TNR), TSS = t(rtestLR3$TSS), HSS = t(rtestLR3$HSS))


dfplot <-  dfplot %>%
  gather(PerformanceMeasure, Score,  ACC:HSS)
ggplot(dfplot, aes(x = p, y = Score, col = PerformanceMeasure, lty = PerformanceMeasure)) + geom_line()+
  geom_vline(xintercept = prev, col = "grey")+ xlim(0, 1)+ylim(0, 1)+ ylab("")+ xlab("p")+ggtitle("Performance Measures")+
  theme(legend.title=element_blank())
# ggsave("ChoosePRDV2.pdf", width = 5, height = 3)

##################################################################################
# apply(rtestLR$TSS, 2, quantile, prob = 0.025)
LR <- apply(rtestLR$TSS, 2, median)
SVM <- apply(rtestSVM$TSS, 2, median)
LR3 <- as.numeric(rtestLR3$TSS)
LR13 <- as.numeric(rtestLR13$TSS)
RF <- as.numeric(rtestRF$TSS)

DNN_8_4 <- as.numeric(rtestDNN_8_4$TSS)
DNN_13_6_6 <- as.numeric(rtestDNN_13_6_6$TSS)
DNN_16_16 <- as.numeric(rtestDNN_16_16$TSS)
DNN_256_32 <- as.numeric(rtestDNN_256_32$TSS)
dfmax <- data.frame(cbind(LR, LR3,  LR13,RF,SVM,DNN_8_4,DNN_16_16, DNN_256_32, DNN_13_6_6 ,p))

xtable(t(t(apply(dfmax[,1:9], 2, max))),digits = 2)



apply(rtestLR$TSS, 2, quantile, prob = 0.025)[20]
apply(rtestLR$TSS, 2, quantile, prob = 0.5)[20]
apply(rtestLR$TSS, 2, quantile, prob = 0.975)[20]

apply(rtestSVM$TSS, 2, quantile, prob = 0.025)[55]
apply(rtestSVM$TSS, 2, quantile, prob = 0.5)[55]
apply(rtestSVM$TSS, 2, quantile, prob = 0.975)[55]


which(LR==max(LR))
apply(rtestLR$HSS, 2, quantile, prob = 0.025)[37]
apply(rtestLR$HSS, 2, quantile, prob = 0.5)[37]
apply(rtestLR$HSS, 2, quantile, prob = 0.975)[37]


apply(rtestSVM$HSS, 2, quantile, prob = 0.5)==max(apply(rtestSVM$HSS, 2, quantile, prob = 0.5))

apply(rtestSVM$HSS, 2, quantile, prob = 0.025)[75]
apply(rtestSVM$HSS, 2, quantile, prob = 0.5)[75]
apply(rtestSVM$HSS, 2, quantile, prob = 0.975)[75]
LR <- apply(rtestLR$HSS, 2, median)
SVM <- apply(rtestSVM$HSS, 2, median)
LR3 <- as.numeric(rtestLR3$HSS)
LR13 <- as.numeric(rtestLR13$HSS)
RF <- as.numeric(rtestRF$HSS)

DNN_8_4 <- as.numeric(rtestDNN_8_4$HSS)
DNN_13_6_6 <- as.numeric(rtestDNN_13_6_6$HSS)
DNN_16_16 <- as.numeric(rtestDNN_16_16$HSS)
DNN_256_32 <- as.numeric(rtestDNN_256_32$HSS)
dfmax <- data.frame(cbind(LR, LR3,  LR13,RF,SVM,DNN_8_4,DNN_16_16, DNN_256_32, DNN_13_6_6 , p))

xtable(t(t(apply(dfmax[,1:9], 2, max))),digits = 2)
apply(rtestLR$TSS, 2, quantile, prob = 0.025)[16]




