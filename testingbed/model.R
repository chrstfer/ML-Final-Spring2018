#rm(list=ls())
set.seed(1)

## Keeping everything else the same as toy example above, except for
## this stuff right here

normalize <- function(x){return(x/sum(x))}
activate <- function(node)
    return(matrix(1/(1+exp(-node))))
sigprime <- function(node)
    return(matrix(activate(node)*(1 - activate(node))))

learningrate <- 0.25
.lengths <- c(784, 16, 4, 10)
truth <- matrix(c(0,0,0,0,0,1,0,0,0,0), ncol=1) # 5 (index by 1 means digits[1]=0)
errs <- list()


model <- list()
.nlayers <- length(.lengths)

.input <- as.vector(tr.images[[1]])
.nodes <- mapply(matrix,
                 data=lapply(.lengths,rnorm, sd=.37),
                 ncol=1,
                 nrow=.lengths)
.weights <-  lapply(1:(.nlayers-1),
                    function(k) {
                        matrix(rnorm(.lengths[k+1]*.lengths[k]),
                               nrow=.lengths[k+1],
                               ncol=.lengths[k])})
.biases <- mapply(matrix,
                  data=lapply(.lengths[-1], rnorm),
                  ncol=1,
                  nrow=.lengths[-1])

## Training
for(n in 1:1000) {
    ## Feed Forward
    model$nodes[[1]] <- model$input
    model$active[[1]] <- activate(model$nodes[[1]])
    
                                        # loop through the rest
    for(i in 2:.nlayers) {
        model$nodes[[i]] <- model$weights[[i-1]] %*% model$active[[i-1]] + model$biases[[i-1]]
        model$active[[i]] <- activate(model$nodes[[i]])
    }
    
                                        # record error of feed forward
    .err <- (model$active[[.nlayers]] - truth)
    print(.err)
    errs[[n]] <- .err
    
                                        # Backprop
    del <- list()
    del[[(.nlayers-1)]] <- (model$active[[.nlayers]] - truth) * sigprime(model$nodes[[.nlayers]])
    ## Loop throught the rest
    for(i in seq((.nlayers-1),2,-1)) {
        del[[i-1]] <- (t(model$weights[[i]]) %*% del[[i]]) * sigprime(model$nodes[[i]])
    }       
    
                                        # Update                                        
    .new.w <- list()
    for(i in length(model$weights):1) {
        .new.w[[i]] <- model$weights[[i]] - learningrate * del[[i]] %*% t(model$active[[i]])
    }
    model$weights <- .new.w
    .new.b <- list()
    for(i in length(model$weights):1) {
        .new.b[[i]] <- model$biases[[i]] - learningrate * del[[i]]
    }
    model$biases <- .new.b
}

## Prediction
which.max(as.vector(model$active[[.nlayers]])) - 1
## Results
errs[[1000]]
model$active[[.nlayers]]
## Error after training
sum((model$active[[4]] - truth)^2)



