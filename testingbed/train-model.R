library(sigmoid)
getLabels <- function(lbldb, nget=0) {
    magic  <- readBin(lbldb, what="integer", n=1, endian="big", size=4)
    if(magic != 2049)
        return(NULL)
    n.lbls    <- readBin(lbldb, what="integer", n=1,    endian="big", size=4)
    if(nget==0)
        nget=n.lbls

    labels <- readBin(lbldb, what="integer", n=nget, endian="big",  size=1)

    close(lbldb)
    return(labels)
}
getImages <- function(imgdb, nget=0, progress=FALSE) {
    magic  <- readBin(imgdb, what="integer", n=1, endian="big", size=4)
    ## if(magic != 2049)
    ##     return(NULL)

    n.imgs <- readBin(imgdb, what="integer", n=1, endian="big", size=4)
    if(nget==0)
        nget <- n.imgs # trunc(sqrt(n.imgs))

    n.rows <- readBin(imgdb, what="integer", n=1, endian="big", size=4)
    n.cols <- readBin(imgdb, what="integer", n=1, endian="big", size=4)

    print(gettextf("Getting %d %dx%d Images", nget, n.rows, n.cols))

    images <- c()
    for(i in 1:nget) {
        .img   <- matrix(readBin(imgdb, what="integer", n=n.rows*n.cols, endian="big", size=1), nrow=n.rows, ncol=n.cols)
        images <-  c(images, list(.img))
        if(progress && i %% trunc(sqrt(nget)) == 0) 
            print(gettextf("%2.2f%%", round((100*i)/nget, digits=2)))
    }
    close(imgdb)
    return(images)
}

## Works
indata.labels <- as.vector(getLabels(gzfile("../datasets/training/labels", "rb"), nget=256))
indata.images <- getImages(gzfile("../datasets/training/images", # data's filename
                                  "rb"), # read it as binary
                           ## Get 256 of the entries
                           nget=256, progress=TRUE)
indata.im.matrix <- do.call("rbind", # create rows out of the input data
                            lapply(indata.images, as.vector)) # transform each image
                                        # matrix into a vector

indata <- cbind(as.factor(indata.labels), indata.im.matrix) # now create a data frame

ts.idx <- sample(length(indata.labels),2*sqrt(length(indata.labels))) ## 4:1 tr:ts (for now)

ts.df <- indata[ts.idx,]
ts.im <- ts.df[-1,]
ts.lb <- ts.df[1,]
ts.images <- indata.images[ts.idx]
ts.labels <- indata.labels[ts.idx]

tr.df <- indata[-ts.idx,]
tr.im <- tr.df[,-1]
tr.lb <- tr.df[,1]
tr.images <- indata.images[-ts.idx]
tr.labels <- indata.labels[-ts.idx]

## basic layout, 4 layers of 5x1
set.seed(1)


ann <- function(node_lengths,
                dlist, l,
                tr.idx, ts.idx,
                learningrate) {
    
    tr.d <- dlist[tr.idx]
    tr.l <- l[tr.idx]
    ts.d <-  dlist[ts.idx]
    ts.l <-  l[ts.idx] 

    model <- new.env()
    model$lengths <- node_lengths
    lengths <- model$lengths
    model$nlayers <- length(model$lengths)
    nlayers <- model$nlayers

    normalize <- function(x){return(x/sum(x))}
    activate <- function(node)
        return(matrix(1/(1+exp(-node))))
    sigprime <- function(node)
        return(matrix(activate(node)*(1 - activate(node))))
    
    model$debug <- TRUE
    model$errs <- list()
    model$biases <- mapply(matrix,
                           data=lapply(lengths[-1], rnorm),
                           ncol=1,
                           nrow=lengths[-1])
    model$weights <- lapply(1:(nlayers-1),
                            function(k) {
                                matrix(rnorm(lengths[k+1]*lengths[k]),
                                       nrow=lengths[k+1],
                                       ncol=lengths[k])})

    ## semi-Pure function: references but does not modify its parent env
    model$predict <- function(input) {
        active <- list()
        nodes <- list()

        nodes[[1]] <- as.vector(input)
        active[[1]] <- activate(nodes[[1]])

        for(i in 2:nlayers) {
            nodes[[i]] <- (model$weights[[i-1]] %*% active[[i-1]])#  + biases[[i-1]])
            active[[i]] <- activate(nodes[[i]])
        }
                
        which.max(as.vector(active[[nlayers]]))-1
    }
        
    train <- function(input, label) {
        truth <- rep(0,10)
        truth[label+1] <- 1

        active <- list()
        nodes <- list()

        nodes[[1]] <- as.vector(input)
        active[[1]] <- activate(nodes[[1]])

        for(i in 2:nlayers) {
            nodes[[i]] <- (weights[[i-1]] %*% active[[i-1]]) #  + biases[[i-1]]
            active[[i]] <- activate(nodes[[i]])
        }
        
        del <- list()
        del[[(nlayers - 1)]] <-  (active[[nlayers]] - truth)* sigprime(nodes[[nlayers]])
        ## n-1, n-2, .. 3, 2
        for(i in seq((nlayers-1), 2, -1)) { 
            del[[i-1]] <- (t(weights[[i]]) %*% del[[i]]) * sigprime(nodes[[i]])
        }
                    
        model$weights <<- lapply(1:length(weights),
                           function(i)
                               return(weights[[i]] - learningrate * del[[i]] %*% t(active[[i]])))

        ## model$biases <<- lapply(1:length(biases),
        ##                   function(i)
        ##                       return(biases[[i]] - learningrate * del[[i]]))
            
        
    }

    
    test <- function(inputs, labels) {        
        preds <- lapply(inputs,model$predict)
        preds==labels
    }

    ## Impure functions
    environment(train) <- model ## MODIFIES ENV
    environment(model$predict) <- model ## Does not modify env   
    
    ## Do initialization
    model$trained <- mapply(train, tr.d, tr.l)
    model$tested <- test(ts.d, ts.l)

    return(model)
}




models <- lapply(seq(0.5,1,0.05),
       ann,
       node_lengths=c(784,16,4,10),
       dlist=indata.images,
       l=indata.labels,
       tr.idx=-ts.idx,
       ts.idx=ts.idx)

lapply(models, function(model) length(which(model$tested)))
