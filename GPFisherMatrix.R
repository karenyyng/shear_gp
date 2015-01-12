# Author: Michael Schneider <mischnei@gmail.com>
# R script to find the ML starting location for a GP process 

library(methods)
library(MASS)
library(mvtnorm)

GPFisherMatrix <- 
  setRefClass("GPFisherMatrix",
               fields=list(des="matrix",
                           rho="numeric",
                           lambda="numeric",
                           nd="numeric",
                           ptheta="numeric",
                           theta_dist="list",
                           prior_lambda_a="numeric",
                           prior_lambda_b="numeric",
                           prior_rho_a="numeric",
                           prior_rho_b="numeric"),
               methods=list(
                 initialize = function(des, rho, lambda) {
                   .self$des = des
                   .self$rho = rho
                   .self$lambda = lambda
                   .self$nd = nrow(des)
                   .self$ptheta = length(rho)
                   .self$theta_dist = get_theta_dist()
                   # Prior hyperparameters
                   .self$prior_lambda_a = 100 #5
                   .self$prior_lambda_b = 1 # 5
                   .self$prior_rho_a = 1
                   .self$prior_rho_b = 0.2
               },
               #
               # Precompute the squared distances between design points
               #
               # Inputs:
               #  	d -- matrix of design points (nd x ptheta)
               #
               get_theta_dist = function() {
                 n <- length(des[,1])
                 
                 inds <- n * (n - 1) / 2
                 indi <- array(0, dim=c(inds,1))
                 indj <- array(0, dim=c(inds,1))
                 
                 ind <- 1
                 for (ii in 1:(n-1)) {
                   indi[ind:(ind+n-ii-1)]=ii
                   indj[ind:(ind+n-ii-1)]=(ii+1):n
                   ind <- ind+n-ii
                 }
                 indm <- indi+n*(indj-1)
                 dist <- (des[indi,] - des[indj,])^2
                 return(list(n=n,indi=indi,indj=indj,indm=indm,d=dist))
               },
               #
               # Compute the Gaussian Process correlation matrix
               #
               # initialize() must be called first to generate theta_dist
               #
               corrmat = function() {
                 # copied from GPM/SA code by James R. Gattiker, 
                 # Los Alamos National Laboratory
                 N <- theta_dist$n
                 R <- matrix(0, nrow=N, ncol=N)
                 beta <- -4*log(rho)  
                 R[theta_dist$indm] <- exp(-beta%*%t(theta_dist$d))
                 R <- R + t(R)
                 diag(R) <- 1
                 return(R)
               },
               # 
               # GP covariance matrix
               #
               covmat = function() {
                 return(corrmat() / lambda + diag(1.e-10, nrow=theta_dist$n))
               },
               #
               # Derivative of the GP corr. matrix with respect to rho_i
               #
               dcorrmat_drho_i = function(i) {
                 theta_i = matrix(0, nrow=nd, ncol=nd)
                 theta_i[theta_dist$indm] <- theta_dist$d[,i]
                 theta_i <- theta_i + t(theta_i)
                 diag(theta_i) <- 0
                 return((4 / rho[i]) * (theta_i %*% corrmat()))
               },
               #
               # Fisher number for the precision parameter
               #
               precision_fisher_matrix = function() {
                 return(0.5 / lambda^2)
               },
               #
               # Fisher matrix for precision and correlation parameters
               #
               fisher_matrix = function() {
                 Sigma_inv = ginv(corrmat())
                 Fmat <- matrix(0, nrow=ptheta, ncol=ptheta)
                 Fcross <- rep(0, ptheta)
                 for (i in 1:ptheta) {
                   dSigma_drho_i = dcorrmat_drho_i(i)
                   Fcross[i] <- 0.5 * sum(diag(dSigma_drho_i %*% Sigma_inv)) / lambda
                   for (j in i:ptheta) {
                     dSigma_drho_j = dcorrmat_drho_i(j)
                     Fmat[i,j] <- sum(diag(Sigma_inv %*% dSigma_drho_i %*%
                                             Sigma_inv %*% dSigma_drho_j))
                     Fmat[j,i] <- Fmat[i,j]                                      
                   }
                 }
                 Ftotal <- matrix(0, nrow=(1+ptheta), ncol=(1+ptheta))
                 Ftotal[1,1] = precision_fisher_matrix()
                 Ftotal[1, 2:(1+ptheta)] = Fcross
                 Ftotal[2:(1+ptheta), 1] = Fcross
                 Ftotal[2:(1+ptheta), 2:(1+ptheta)] <- Fmat
                 return(Ftotal)
               },
               #
               # ML estimator for the GP precision given centered 
               # observations at the design points, delta.
               #
               precision_quad_estimator = function(delta) {
                 return(1 / sum(t(delta) %*% ginv(corrmat()) %*% delta))
               },
               #
               # ML estimator for the GP correlation parameters given
               # centered observations at the design points, delta.
               #
               correlation_quad_estimator = function(delta) {
                 Fisher <- fisher_matrix()[2:(1+ptheta), 2:(1+ptheta)]
                 F_inv <- ginv(Fisher)
                 Cmat <- covmat()
                 Cmat_inv <- ginv(Cmat)
                 term1 <- rep(0, ptheta)
                 term2 <- rep(0, ptheta)
                 for (i in 1:ptheta) {
                   dC_drho_i <- dcorrmat_drho_i(i) / lambda
                   term1[i] <- sum(t(delta) %*% Cmat_inv %*% dC_drho_i %*% Cmat_inv %*% delta)
                   term2[i] <- sum(diag(Cmat_inv %*% dC_drho_i))
                 }
                 print(term1)
                 print(term2)
                 
                 est <- F_inv %*% (term1 - term2) / 2
                 return(est)
               },
               #
               # Prior on precision parameter
               #
               ln_prior_lambda = function() {
                 return(dgamma(lambda, shape=prior_lambda_a,
                               rate=prior_lambda_b, log=TRUE))
               },
               # 
               # Prior on correlation parameters
               #
               ln_prior_rho = function() {
                 return(sum(dbeta(rho, prior_rho_a, prior_rho_b, log=TRUE)))
               },
               #
               # GP log-likelihood
               #
               ln_likelihood = function(y) {
                 lnL <- dmvnorm(y, rep(0, length(y)), covmat(), log=TRUE)
                 return(lnL)
               },
               #
               # Gradient of GP log-likelihood
               #  Derivatives are taken with respect to:
               #    ln(lambda)
               #    beta_i := -4 ln(rho_i)
               #
               ln_likelihood_grad = function(y) {
                 Cmat_inv <- ginv(covmat())
                 term1 <- Cmat_inv %*% y %*% t(y) - diag(1, nd)
                 grad <- rep(0, 1 + ptheta)
                 grad[1] <- -0.5 * sum(diag(term1))
                 #
                 for (i in 1:ptheta) {
                   theta_i = matrix(0, nrow=nd, ncol=nd)
                   theta_i[theta_dist$indm] <- theta_dist$d[,i]
                   theta_i <- theta_i + t(theta_i)
                   diag(theta_i) <- 0
                   grad[i+1] <- -0.5 * sum(diag(term1 %*% theta_i))
                 }
                 return(grad[2:(1+ptheta)])
               },
               # 
               # Gradient of GP correlation prior
               #
               ln_prior_rho_grad = function() {
                 x <- rho^(prior_rho_b-1) 
                 term2 <- (prior_rho_b - 1) * x / (1 - x) 
                 p <- log((1 - prior_rho_a + term2)/4)
                 return(p + ln_prior_rho())
               }
               ))

find_ML_GP_params <- function(des, y, verbose=TRUE) {
  lambda <- 0.1
  rho <- rep(0.9, ncol(des))  # correlation parameters
  beta <- -4 * log(rho)  # inverse length parameters
  xx <- GPFisherMatrix$new(des=as.matrix(des), rho=rho, lambda=lambda)

  # Evaluate the likelihood for new parameters p
  fn <- function(p, y) {
    xx$lambda <- exp(p[1])
    xx$rho <- exp(-p[2:(1 + xx$ptheta)] / 4)
#     print(xx$lambda)
#     print(xx$rho)
    lnL <- xx$ln_likelihood(y)
    p_rho <- xx$ln_prior_rho() + log(abs(-4 / prod(rho)))
    p_lambda <- xx$ln_prior_lambda() +  log(abs(lambda))
#     print(lnL)
    return(lnL + p_rho + p_lambda)
  }

  # Evaluate the gradient of the likelihood for new parameters p
  gr <- function(p, y) {
#     xx$lambda <- exp(p[1])
    xx$rho <- exp(-p / 4)
    dlnL <- xx$ln_likelihood_grad(y)
    dprho <- xx$ln_prior_rho_grad()
    return(dlnL + dprho)
  }

  # Optimization routine
  #  Supplying the actual gradient function causes convergence failure.
  #  optim appears to work if the gradient is estimated internally instead.
  res <- optim(par=c(log(lambda), beta), fn=fn, y=y, method="L-BFGS-B",
               lower=c(-10, rep(4e-4, xx$ptheta)),
               upper=c(10, rep(28, xx$ptheta)),
               control=list(trace=6, fnscale=-1))
  print(res)
  gp.params <- c(exp(res$par[1]),
                 exp(-0.25 * res$par[2:(1 + length(rho))]))
  if (verbose) {
    cat(sprintf("lambda: %3.2g\n", exp(res$par[1])))
    cat(sprintf("rho:"))
    cat(sprintf("%5.4g", exp(-0.25 * res$par[2:(1 + length(rho))])))
    cat(sprintf("\n"))
  }
#   print(exp(-0.25 * res$par))
  return(gp.params)
}
