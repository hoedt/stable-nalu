rm(list = ls())
setwd(dirname(parent.frame(2)$ofile))

library(plyr)
library(dplyr)

simulate.mse = function (epsilon, samples, operation, simple, input.size, seq.length, subset.ratio, overlap.ratio, range.a, range.b, range.mirror) {
  X = matrix(runif(samples*input.size*seq.length, range.a, range.b), samples * seq.length, input.size)

  if (range.mirror) {
    X.sign = matrix(rbinom(samples*input.size * seq.length, 1, 0.5), samples * seq.length, input.size) * 2 - 1
    X = X * X.sign
  }

  evaluate = function (W, U) {
    Sx = X %*% W
    Z = matrix(0, samples, dim(W)[2])
    for (i in 1:seq.length) {
      Z = Sx[((i-1)*samples + 1):(i*samples),] + Z %*% U
    }

    if (operation == 'add') {
      Y = Z[,1] + Z[,2]
    } else if (operation == 'sub') {
      Y = Z[,1] - Z[,2]
    } else if (operation == 'mul') {
      Y = Z[,1] * Z[,2]
    } else if (operation == 'div') {
      Y = Z[,1] / Z[,2]
    } else if (operation == 'squared') {
      Y = Z[,1] * Z[,1]
    } else if (operation == 'root') {
      Y = sqrt(Z[,1])
    }
    return(Y)
  }
  
  target.matrix = function (epsilon) {
    if (simple) {
      a.start = 1
      a.end = 2
      b.start = 1
      b.end = 4
    } else {
      subset.size = floor(subset.ratio * input.size)
      overlap.size = floor(overlap.ratio * subset.size)
      
      a.start = 1
      a.end = a.start + subset.size
      b.start = a.end - overlap.size
      b.end = b.start + subset.size
    }
    
    W = matrix(0 + epsilon, input.size, 2)
    W[a.start:a.end, 1] = 1 - epsilon
    W[b.start:b.end, 2] = 1 - epsilon
    return(W)
  }

  W.y = target.matrix(epsilon)
  U.y = (1 - epsilon) * diag(2)
  W.t = target.matrix(0)
  U.t = diag(2)
  errors = (evaluate(W.y, U.y) - evaluate(W.t, U.t))**2
  
  return(mean(errors))
}

cases = rbind(
  c(parameter='default', operation='add', simple=F, input.size=10, seq.length=100, subset.ratio=0.25, overlap.ratio=0.5, range.a=1, range.b=2, range.mirror=F),
  c(parameter='default', operation='sub', simple=F, input.size=10, seq.length=100, subset.ratio=0.25, overlap.ratio=0.5, range.a=1, range.b=2, range.mirror=F),
  c(parameter='default', operation='mul', simple=F, input.size=10, seq.length=100, subset.ratio=0.25, overlap.ratio=0.5, range.a=1, range.b=2, range.mirror=F),
  c(parameter='range', operation='add', simple=F, input.size=10, seq.length=10, subset.ratio=0.25, overlap.ratio=0.5, range.a=2, range.b=6, range.mirror=F),
  c(parameter='range', operation='sub', simple=F, input.size=10, seq.length=10, subset.ratio=0.25, overlap.ratio=0.5, range.a=2, range.b=6, range.mirror=F),
  c(parameter='range', operation='mul', simple=F, input.size=10, seq.length=10, subset.ratio=0.25, overlap.ratio=0.5, range.a=2, range.b=6, range.mirror=F)
)

for (input.size in c(10, 100, 1000)) {
  cases = rbind(cases, c(parameter='input.size', operation='add', simple=F, input.size=input.size, seq.length=10, subset.ratio=0.25, overlap.ratio=0.5, range.a=1, range.b=2, range.mirror=F))
}

for (seq.length in c(10, 20, 1000)) {
  cases = rbind(cases, c(parameter='seq.length', operation='add', simple=F, input.size=10, seq.length=seq.length, subset.ratio=0.25, overlap.ratio=0.5, range.a=1, range.b=2, range.mirror=F))
  cases = rbind(cases, c(parameter='seq.length', operation='sub', simple=F, input.size=10, seq.length=seq.length, subset.ratio=0.25, overlap.ratio=0.5, range.a=1, range.b=2, range.mirror=F))
  cases = rbind(cases, c(parameter='seq.length', operation='mul', simple=F, input.size=10, seq.length=seq.length, subset.ratio=0.25, overlap.ratio=0.5, range.a=1, range.b=2, range.mirror=F))
}

eps = data.frame(rbind(
  c(operation='mul', epsilon=0.00001),
  c(operation='add', epsilon=0.00001),
  c(operation='sub', epsilon=0.00001),
  c(operation='div', epsilon=0.00001)
))

mse = data.frame(cases) %>%
  merge(eps) %>%
  mutate(
    simple=as.logical(as.character(simple)),
    input.size=as.integer(as.character(input.size)),
    seq.length=as.integer(as.character(seq.length)),
    subset.ratio=as.numeric(as.character(subset.ratio)),
    overlap.ratio=as.numeric(as.character(overlap.ratio)),
    range.a=as.numeric(as.character(range.a)),
    range.b=as.numeric(as.character(range.b)),
    range.mirror=as.logical(as.character(range.mirror)),
    epsilon=as.numeric(as.character(epsilon))
  ) %>%
  rowwise() %>%
  mutate(
    threshold=simulate.mse(epsilon, 10000, operation, simple, input.size, seq.length, subset.ratio, overlap.ratio, range.a, range.b, range.mirror),
    extrapolation.range=ifelse(range.mirror, paste0('U[-',range.b,',-',range.a,'] ∪ U[',range.a,',',range.b,']'), paste0('U[',range.a,',',range.b,']')),
    operation=paste0('op-', operation)
  )

write.csv(mse, file="../results/function_task_recurrent_mse_expectation.csv", row.names=F)

