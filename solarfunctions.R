####################
# Useful functions #
####################

is.even <- function(x) x %% 2 == 0

logitInv <- function(z) 1 / (1 + exp( - z))

funcLogical  <- function(v1, v2) matrix(c(sum(!v1 & !v2), sum(v1 & !v2), sum(!v1 & v2), sum(v1 & v2)),2,2)
  

calcSkillScore <- function(yhat, y)
{
  r <- new.env()
  
  result.table <- funcLogical(yhat, y)
  
  TN <- as.numeric(result.table[1, 1]) # to avoid integer overflow: as.numeric
  TP <- as.numeric(result.table[2, 2])
  FN <- as.numeric(result.table[1, 2])
  FP <- as.numeric(result.table[2, 1])
  
  TPR <- TP/sum(result.table[ , 2])
  TNR <- TN/sum(result.table[ , 1])
  FPR <- FP/sum(result.table[ , 1])
  FNR <- FN/sum(result.table[ , 2])
  TSS <- TP/(TP + FN) - FP/(FP + TN)
  HSS <- 2 * ((TP * TN) - (FN * FP))/((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN))
  ACC <- (TP + TN)/(TP + FN + FP + TN)
  r$TSS <- TSS
  r$TPR <- TPR
  r$TNR <- TNR
  r$FPR <- FPR
  r$FNR <- FNR
  r$HSS <- HSS
  r$ACC <- ACC
  r <- as.list(r)
  return(r)
}
  
getSkillScore <- function(p = seq(0.1, 0.9, by = 0.1),  pHat, y, subsets = NULL)

{
  r <- new.env()

  if(is.null(dim(pHat)))nResamp <- 1
  else nResamp <- dim(pHat)[2]
  
  pLen <- length(p)
  TPR <- matrix(0, nResamp, pLen)
  FPR <- matrix(0, nResamp, pLen)
  FNR <- matrix(0, nResamp, pLen)
  TNR <- matrix(0, nResamp, pLen)
  TSS <- matrix(0, nResamp, pLen)
  HSS <- matrix(0, nResamp, pLen)
  ACC <- matrix(0, nResamp, pLen)

  # if testset
  if(is.null(subsets))y <- as.numeric(y) 
  else yTemp <- as.numeric(y)

  for(j in 1 : nResamp){
      if(!is.null(subsets))y <- yTemp[-subsets[j, ]]

      for(i in 1 : pLen){
        yhat <- as.numeric(pHat[, j] > p[i])
        skill <- calcSkillScore(yhat, y )

        TPR[j, i] <- skill$TPR
        TNR[j, i] <- skill$TNR
        FPR[j, i] <- skill$FPR
        FNR[j, i] <- skill$FNR
        TSS[j, i] <- skill$TSS
        HSS[j, i] <- skill$HSS      
        ACC[j, i] <- skill$ACC
     }
  }

  r$TSS <- TSS
  r$TPR <- TPR
  r$TNR <- TNR
  r$FPR <- FPR
  r$FNR <- FNR
  r$HSS <- HSS
  r$ACC <- ACC
  r$p <- p
  r <- as.list(r)
  class(r) = "SkillScores"
  return(r)
}







ltxtable <- function(r, q1 = 0.025, q2 = 0.975){
  
  TSS <- r$TSS
  TPR <- r$TPR
  TNR <- r$TNR
  FPR <- r$FPR
  HSS <- r$HSS
  ACC <- r$ACC
  p <- r$p

  HSSmed <- apply(HSS, 2, median)
  ACCmed <- apply(ACC, 2, median)
  HSSL <- apply(HSS, 2, quantile, prob = q1)
  ACCL <- apply(ACC, 2, quantile, prob = q1)
  HSSU <- apply(HSS, 2, quantile, prob = q2)
  ACCU <- apply(ACC, 2, quantile, prob = q2)
  
  
  TSSmed <- apply(TSS, 2, median)
  TSSL <- apply(TSS, 2, quantile, prob = q1)
  TSSU <- apply(TSS, 2,  quantile, prob = q2)
  
  TPRmed <- apply(TPR, 2, median)
  TPRL <- apply(TPR, 2, quantile, prob = q1)
  TPRU <- apply(TPR, 2,  quantile, prob = q2)
 
  TNRmed <- apply(TNR, 2, median)
  TNRL <- apply(TNR, 2, quantile, prob = 0.025)
  TNRU <- apply(TNR, 2,  quantile, prob = 0.975)

  return(cbind(p, TSSmed, TSSL, TSSU,  TPRmed, TPRL, TPRU, TNRmed,TNRL,TNRU, ACCmed,ACCU,ACCL,HSSmed, HSSL, HSSU))
}

 
calcAUC <- function(TPR, FPR){
  # inputs already sorted
  dFPR <- c(diff(FPR), 0)
  dTPR <- c(diff(TPR), 0)
  sum(TPR * dFPR) + sum(dTPR * dFPR)/2
}

getAUC <- function(TPR, FPR){

  nResamp <-  dim(TPR)[1]
  zeros <- rep(0, nResamp)
  ones <- rep(1, nResamp)
  TPR <- cbind(ones, TPR, zeros)
  FPR <- cbind(ones, FPR, zeros)
  TPR <- t(apply(TPR, 1, sort))
  FPR <- t(apply(FPR, 1, sort))
  AUC <- zeros
  for (i in 1: nResamp)AUC[i] <- calcAUC(TPR[i,], FPR[i,])
  return(AUC)
  }

  




subSample <- function(nResamp = 50, n1 = 100, n0 = 300, N1, N0){


  index0 <- matrix(0, nResamp, n0)
  index1 <- matrix(0, nResamp, n1)
  for (i in 1 : nResamp) #resample
  {
    index0[i, ] <- sort(sample(N0, n0))
    index1[i, ] <- sort(sample(N1, n1)) + N0
  }  
  
  indexSet <- cbind(index0, index1)
  return(indexSet)
  
}




