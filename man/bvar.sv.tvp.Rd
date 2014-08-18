\name{bvar.sv.tvp}
\alias{bvar.sv.tvp}
\title{
Bayesian Analysis of a Vector Autoregressive Model with Stochastic Volatility and Time-Varying Parameters
}
\description{
Bayesian estimation of the flexible VAR model by Primiceri (2005) which allows for both stochastic volatility and time drift in the model parameters. 
}
\usage{
bvar.sv.tvp(Y, p = 1, tau = 40, nf = 10, pdrift = TRUE, nrep = 50000, 
nburn = 5000, thinfac = 10, itprint = 10000, k_B = 4, k_A = 4, k_sig = 1, 
k_Q = 0.01, k_S = 0.1, k_W = 0.01, pQ = NULL, pW = NULL, pS = NULL)
}
\arguments{
  \item{Y}{Matrix of data, where rows represent time and columns are different variables. \code{Y} must have at least two columns.}
  \item{p}{Lag length, greater or equal than 1 (the default)}
  \item{tau}{Length of the training sample used for determining prior parameters via least squares}
  \item{nf}{Number of future time periods for which forecasts are computed (integer, 1 or greater, defaults to 10)}
  \item{pdrift}{Dummy, indicates whether or not to account for parameter drift when simulating forecasts (defaults to TRUE).}
  \item{nrep}{Number of MCMC draws excluding burn-in (defaults to 50000)}
  \item{nburn}{Number of MCMC draws used to initialize the sampler (defaults to 5000). These draws do not enter the computation of posterior moments, forecasts etc.}
   \item{thinfac}{Thinning factor for MCMC output. Defaults to 10, which means that the forecast sequences (\code{fc.mdraws}, \code{fc.vdraws}, \code{fc.ydraws}, see below) contain only every tenth draw of the original sequence. Set \code{thinfac} to one to obtain the full MCMC sequence.}
  \item{itprint}{Print every itprint-th iteration. Defaults to 10000. Set to very large value to omit printing altogether.}
  \item{k_B, k_A, k_sig, k_Q, k_W, k_S, pQ, pW, pS}{Quantities which enter the prior distributions, see the vignette for details. Defaults to the exact values used in the original article by Primiceri.}
}
\value{
  \item{Beta.postmean}{Posterior means of coefficients. This is an array of dimension \eqn{[M, Mp+1, T]}, where \eqn{T} denotes the number of time periods (= number of rows of \code{Y}), and \eqn{M} denotes the number of system variables (= number of columns of \code{Y}). The submatrix \eqn{[, , t]} represents the coefficient matrix at time \eqn{t}. The intercept vector is stacked in the first column; the \code{p} coefficient matrices of dimension \eqn{[M,M]} are placed next to it.}
  \item{H.postmean}{Posterior means of error term covariance matrices. This is an array of dimension \eqn{[M, M, T]}. The submatrix \eqn{[, , t]} represents the covariance matrix at time \eqn{t}.}
  \item{Q.postmean, S.postmean, W.postmean}{Posterior means of various covariance matrices.}
  \item{fc.mdraws}{Draws for the forecast mean vector at various horizons (three-dimensional array, where the first dimension corresponds to system variables, the second to forecast horizons, and the third to MCMC draws). \emph{Note:} The third dimension will be equal to \eqn{nrep/thinfac}, apart from possible rounding issues.}
  \item{fc.vdraws}{Draws for the forecast covariance matrix. Design similar to \code{fc.mdraws}, except that the first array dimension contains the lower-diagonal elements of the forecast covariance matrix.}
  \item{fc.ydraws}{Simulated future observations. Design analogous to \code{fc.mdraws}.}
}
\references{
Del Negro, M. and G. Primiceri (2014): ``Time Varying Structural Vector Autoregressions and Monetary Policy: A Corrigendum'', working paper, Northwestern University. 

Koop, G. and D. Korobilis (2010): ``Bayesian Multivariate Time Series Methods for Empirical Macroeconomics'', Foundations and Trends in Econometrics 3, 267-358.

Primiceri, G. (2005): ``Time Varying Structural Vector Autoregressions and Monetary Policy'', Review of Economic Studies 72, 821-852.
}
\author{
Fabian Krueger, based on Matlab code by Dimitris Korobilis (see Koop and Korobilis, 2010). \emph{Incorporates the corrigendum by Del Negro and Primiceri (2014), which points to an error in the original MCMC algorithm.}
}
\examples{
\dontrun{

# Load US macro data
data(usmacro)

# Estimate trivariate BVAR using default settings
bv <- bvar.sv.tvp(usmacro)

}
}
\seealso{For detailed examples and explanations, see the vignette hosted at \url{https://sites.google.com/site/fk83research/code}.}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{forecasting methods}
