#rm(list=ls())
set.seed(1)

activate <- function(node)
    return(matrix(1/(1+exp(-node))))
sigprime <- function(node)
    return(matrix(activate(node)*(1 - activate(node))))

truth <- 1
learningrate <- 0.25
errs <- c()
.lengths <- c(2,3,3,1) # I,H,O


model <- list()
.nlayers <- length(.lengths)
model$input <- matrix(c(1,0))
model$nodes <- mapply(matrix,
		      data=lapply(.lengths,rnorm, sd=.37),
		      ncol=1,
		      nrow=.lengths)
model$weights <-  lapply(1:(.nlayers-1),
			 function(k) {
			     matrix(rnorm(.lengths[k+1]*.lengths[k]),
				    nrow=.lengths[k+1],
				    ncol=.lengths[k])})
model$biases <- mapply(matrix,
		       data=lapply(.lengths[-1], rnorm),
		       ncol=1,
		       nrow=.lengths[-1])

n <- 0
for(n in 1:250) {
    
			    # Feed Forward
    model$nodes[[1]] <- model$input
    model$active[[1]] <- activate(model$nodes[[1]])

    ## loop through the rest
    for(i in 2:.nlayers) {
        model$nodes[[i]] <- model$weights[[i-1]] %*% model$active[[i-1]] + model$biases[[i-1]]
        model$active[[i]] <- activate(model$nodes[[i]])
    }

                                        # record error of feed forward
    errs[n] <- model$active[[.nlayers]] - truth

    
                                        # Backprop
    del <- list()
    del[[(.nlayers-1)]] <- (model$active[[.nlayers]] - truth) * sigprime(model$nodes[[.nlayers]])
    ## Loop throught the rest
    for(i in seq((.nlayers-1),2,-1)) {
        print(i)
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
