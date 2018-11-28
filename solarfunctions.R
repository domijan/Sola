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




plotLines <- function(x, q1, q2, p, skillname){
  xmed <- apply(x, 2, median)
  xL <- apply(x, 2, quantile, prob = q1)
  xU <- apply(x, 2,  quantile, prob = q2)
  df <- data.frame(cbind(xmed, xL, xU, p))
  ggplot(df, aes(x = p, y = xmed)) + geom_line()+
    xlim(0, 1)+ylim(0, 1)+ ylab(skillname)+ geom_ribbon(aes(ymin=xU, ymax=xL), alpha=0.2)
}

plotBoxes <- function(x, p, skillname){
  
  colnames(x) <- p
  df <-melt(x)
  ggplot(df, aes(as.factor(Var2), value)) + geom_boxplot()+ ylab(skillname) +xlab("p")
  
}
plot.SkillScores <- function(r,  type = "default", q1 = 0.025, q2 = 0.975, ...){
 
  TSS <- r$TSS
  TPR <- r$TPR
  TNR <- r$TNR
  FPR <- r$FPR
  HSS <- r$HSS
  ACC <- r$ACC
  p <- r$p

  if(type == "boxplot"){
    plotBoxes(TPR,  p, "TPR")+ylim(0, 1)+ scale_x_discrete(breaks=seq(0,1,by = 0.1))
    plotBoxes(TNR,  p, "TNR")+ylim(0, 1)+ scale_x_discrete(breaks=seq(0,1,by = 0.1))
    plotBoxes(TSS,  p, "TSS")+ylim(0, 1)+ scale_x_discrete(breaks=seq(0,1,by = 0.1))
    ggsave("TSSBox.pdf", width = 3, height = 3)
    plotBoxes(HSS,  p, "HSS")+ylim(0, 1)+ scale_x_discrete(breaks=seq(0,1,by = 0.1))
    ggsave("HSSBox.pdf", width = 3, height = 3)
    plotBoxes(ACC,  p, "ACC")+ylim(0, 1)+ scale_x_discrete(breaks=seq(0,1,by = 0.1))
    ggsave("ACCBox.pdf", width = 3, height = 3)
 
  }

  else if(type == "default"){
    plotLines(TPR, q1, q2, p, "TPR")
    ggsave("TPR.pdf", width = 3, height = 3)
    plotLines(TNR, q1, q2, p, "TNR")
    ggsave("TNR.pdf", width = 3, height = 3)
    plotLines(TSS, q1, q2, p, "TSS")
    ggsave("TSS.pdf", width = 3, height = 3)
    plotLines(HSS, q1, q2, p, "HSS")
    ggsave("HSS.pdf", width = 3, height = 3)
    plotLines(ACC, q1, q2, p, "ACC")
    ggsave("ACC.pdf", width = 3, height = 3)
  }
  
  
  else if(type == "roc"){
    TPR <- cbind(rep(1, nResamp), TPR, rep(0, nResamp))
    FPR <- cbind(rep(1, nResamp), FPR, rep(0, nResamp))
    df <- melt(TPR)
    df$FPR <- melt(FPR)$value
    ggplot(df, aes(x = FPR, y = value, group = Var1)) + geom_line() + ylab("TPR")
    # ggsave("ROC.pdf", width = 3, height = 3)
   }
  else stop("error: Plot type not supported for a SkillScore object")
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




getGrid <- function(x, corse){ 
  
  nf <- dim(x)[2]
  maxim <- matrix( 0, nf, 2)
  sqs <- matrix( 0, nf, corse)
  e <- 1
  e <- as.list(e)
  for(i in 1:nf){
    maxim[i, 1] <- max(x[ ,i])
    maxim[i, 2] <- min(x[ ,i])
    sqs[i, ] <- seq(maxim[i, 2], maxim[i, 1], length.out = corse)
    e[[i]] <- sqs[i, ]
  }
  
  
  grid <- expand.grid(e)
  
  
 names(grid) <- colnames(x)
 return(grid)
}




drawContours <- function(features, corse, beta, labels, ... ){ 
  
  
  grid <- getGrid(features, corse)
  grid$z <- logitInv(as.matrix(cbind(rep(1, dim(grid)[1]), grid)) %*% beta)
  
  
  plot(features[ ,1], features[ ,2], bty = "l", xlab= colnames(features)[1], ylab= colnames(features)[2], col = labels)
  xs <- seq(par("xaxp")[1], par("xaxp")[2],length.out =  par("xaxp")[3] + 1)
  ys <- seq(par("yaxp")[1], par("yaxp")[2],length.out =  par("yaxp")[3] + 1)
  boundingBox <- par("usr")
  rect(boundingBox[1], boundingBox[3], boundingBox[2], boundingBox[4], col = "lightgrey", border=NA)
  abline(v = xs, col = "white", lwd = 0.5)
  abline(h = ys, col = "white", lwd = 0.5)
  axis(1, at = xs,  labels = NULL,  col = "grey")
  axis(2, at = ys,  labels = NULL,  col = "grey")

  points(features[ ,1], features[ ,2], col = labels)

  contour(as.matrix(unique(grid[, 1])), as.matrix(unique(grid[, 2])), matrix(grid$z, corse, corse),
  add = TRUE, ...)

  
}




drawLevels <- function(features, corse, beta, labels, ... ){ 
  

  grid <- getGrid(features, corse)
  grid$z <- logitInv(as.matrix(cbind(rep(1, dim(grid)[1]), grid)) %*% beta)
  


  image(as.matrix(unique(grid[, 1])), as.matrix(unique(grid[, 2])), matrix(grid$z, corse, corse),
                    bty = "l", xlab= colnames(features)[1], ylab= colnames(features)[2],  ...)


  contour(as.matrix(unique(grid[, 1])), as.matrix(unique(grid[, 2])), matrix(grid$z, corse, corse),  col = "grey50",
          add = TRUE)
  points(features[ ,1], features[ ,2], col = labels)
  
}


# drawLevels <- function(features, corse, beta, labels,  ... ){ 
#   
# 
#   grid <- getGrid(features, corse)
#   
#   grid$z <- logitInv(as.matrix(cbind(rep(1, dim(grid)[1]), grid)) %*% beta)
#   
#   names(grid)[1] <- "x" 
#   names(grid)[2] <- "y" 
#   
#   levelplot(z ~ x * y, grid,  xlab= colnames(features)[1], ylab= colnames(features)[2], 
#                  panel = function(x, y, subscripts, ...){
#                    panel.contourplot(x, y, subscripts, ...)
#                    panel.xyplot(features[ ,1], features[ ,2], col = labels)},... )
#   
# 
# }
# 


drawThresholds <- function(features, corse, beta, labels, a = 0.2, nby = 20, colorcontour  = 3, ...){ 
  
  
  grid <- getGrid(features, corse)
  
  nResamp <- dim(beta)[1]
  poorprob3 <- matrix(0, dim(grid)[1], nResamp)
  
  for (i in 1 : nResamp)poorprob3[, i] <- logitInv(as.matrix(cbind(rep(1, dim(grid)[1]), grid)) %*% beta[i, ])

 

  
  plot(features[ ,1], features[ ,2], bty = "l",
       xlab = colnames(features)[1], ylab= colnames(features)[2], col = labels, ...)           
  xs <- seq(par("xaxp")[1], par("xaxp")[2],length.out =  par("xaxp")[3] + 1)
  ys <- seq(par("yaxp")[1], par("yaxp")[2],length.out =  par("yaxp")[3] + 1)
  boundingBox <- par("usr")
  rect(boundingBox[1], boundingBox[3], boundingBox[2], boundingBox[4], col = "lightgrey", border=NA)
  abline(v = xs, col = "white", lwd = 0.5)
  abline(h = ys, col = "white", lwd = 0.5)
  axis(1, at = xs,  labels = NULL,  col = "grey")
  axis(2, at = ys,  labels = NULL,  col = "grey")
  points(features[ ,1], features[ ,2], col = labels)

  
  
  if (length(colorcontour) ==1) colorcontour <- rep(colorcontour,nResamp/nby ) #checkok
  for(i in seq(1, nResamp, by = nby))contour(as.matrix(unique(grid[, 1])), as.matrix(unique(grid[, 2])), 
                                             matrix(poorprob3[,i] , corse, corse),
                                          add = TRUE, levels = a, col = colorcontour[i])
}


