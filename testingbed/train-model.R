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
tr.labels <- as.vector(getLabels(gzfile("../datasets/training/labels", "rb"), nget=256))
tr.images <- getImages(gzfile("../datasets/training/images", # data's filename
                              "rb"), # read it as binary
                       ## Get 256 of the entries
                       nget=256, progress=TRUE)
tr.im.matrix <- do.call("rbind", # create rows out of the input data
                        lapply(tr.images, as.vector)) # transform each image
                                        # matrix into a vector

tr.df <- cbind(as.factor(tr.labels), tr.im.matrix) # now create a data frame

model.gen.annc <- function(layers,
                           training.data,
                           training.labels,
			   debug=TRUE) {    
    .lengths <- layers    
    .nlayers <- length(.lengths)

    if(debug) print(paste("Number of Layers:", n.layers))
    if(debug) print(paste("Layer Lengths:",    toString(.lengths)))
    
    model <- train(training.data, training.labels)
    class(model) <- "model.ann.classifier"
    return(model) 
}


normalize <- function(x){return(x/sum(x))}
activate <- function(node)
    return(matrix(1/(1+exp(-node))))
sigprime <- function(node)
    return(matrix(activate(node)*(1 - activate(node))))


train <- function(data, labels) { # row-wise dataframe and list
    .lengths <- c(784,16,8,10)
    .nlayers <- length(.lengths)
    errs <- list()
    
    .active <- mapply(matrix,
                      data=1,
                      ncol=1,
                      nrow=.lengths)
    .nodes <- mapply(matrix,
                     data=1,
                     ncol=1,
                     nrow=.lengths)
    .biases <- mapply(matrix,
                      data=lapply(.lengths, rnorm),
                      ncol=1,
                      nrow=.lengths)
    .weights <- lapply(1:(.nlayers-1),
                       function(k) {
                           matrix(rnorm(.lengths[k+1]*.lengths[k]),
                                  nrow=.lengths[k+1],
                                  ncol=.lengths[k])})

    

    
    ## lapply(tr.df[,-1],thefollowing) <- do once for each and then average?
    ## split whole db into batches, find average across
    for(n in 1:nrow(data)) {
        truth <- rep(0,10)
        truth[labels[n]+1] <- 1
        
        .nodes[[1]] <- matrix(data[n,],ncol=1)
        .active[[1]] <- activate(.nodes[[1]])
        ## We could definitely get this faster. Each iteration only depends
        ## on the previous layer's active, if we could keep passing that
        ## down the chain
        .weights[[1]] %*% .active[[1]]

        for(i in 2:(.nlayers)) {
            print(i)
            setBreakpoint(srcfile="train-model.R",line=112)
                                        #print(dim(.weights[[i-1]]))
                                        #print(dim(.active[[i-1]]))                
            ##.nodes[[i]] <- .active[[i-1]] %*% .weights[[i-1]] + .biases[[i-1]]               
            .nodes[[i]] <- (.weights[[i-1]] %*% .active[[i-1]])  + .biases[[i-1]]
            .active[[i]] <- activate(.nodes[[i]])
        }

        if(debug)
            errs[[n]] <- (model$active[[.nlayers]] - truth)

        del <- list()
        del[[(.nlayers-1)]] <- (.active[[.nlayers]] - truth) * sigprime(.nodes[[.nlayers]])
        ## Loop throught the rest
        for(i in seq((.nlayers-1),2,-1)) {
            del[[i-1]] <- (t(.weights[[i]]) %*% del[[i]]) * sigprime(.nodes[[i]])
        }

        .weights <- lapply(length(.weights):1,
                           function(i)
                               return(.weights[[i]] - learningrate * del[[i]] %*% t(.active[[i]])))
        
        .biases <- lapply(length(.weights):1, 
                          function(i)
                              return(.biases[[i]] - learningrate * del[[i]]))
    }
    
    trained <- list()
    trained$nodes <- .nodes
    trained$weights <- .weights
    trained$biases <- .biases

    trained$nlayers <- n.layers
    trained$lengths <- .lengths
    
    return(trained)
}



## basic layout, 4 layers of 5x1
set.seed(1)



r1 <- predict(model, c(1,0))



r2 <- predict(model, c(1,1))
r3 <- predict(model, c(0,1))
r4 <- predict(model, c(0,0))

print("Results ::")
paste("Results 1: ",r1)
paste("Results 1: ",r2)
paste("Results 1: ",r3)
paste("Results 1: ",r4)
