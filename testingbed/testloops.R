v <- seq(1,3)

.lengths <- c(2,2,1)
s <- list()
s$nodes <- mapply(matrix,
                  data=(lapply(.lengths, rnorm)),
                  nrow=.lengths,
                  ncol=1)

update <- function(model, k) {
    model$nodes[[k]] <- k*model$nodes[[k]]
    return(model)
}

.backup <- s
print(s$nodes)
sumlist <- lapply(v, function(i){
    s <- update(s, i)
    return(s)
})
lapply(1:3,function(i)print(s$nodes[[i]]==backup$nodes[[i]]))
