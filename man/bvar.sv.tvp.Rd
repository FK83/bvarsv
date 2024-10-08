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
nburn = 5000, thinfac = 10, itprint = 10000, save.parameters = TRUE, 
k_B = 4, k_A = 4, k_sig = 1, k_Q = 0.01, k_S = 0.1, k_W = 0.01, 
pQ = NULL, pW = NULL, pS = NULL)
}
\arguments{
  \item{Y}{Matrix of data, where rows represent time and columns are different variables. \code{Y} must have at least two columns.}
  \item{p}{Lag length, greater or equal than 1 (the default)}
  \item{tau}{Length of the training sample used for determining prior parameters via least squares (LS). That is, data in \code{Y[1:tau, ]} are used for estimating prior parameters via LS; formal Bayesian analysis is then performed for data in \code{Y[(tau+1):nrow(Y), ]}.}
  \item{nf}{Number of future time periods for which forecasts are computed (integer, 1 or greater, defaults to 10).}
  \item{pdrift}{Dummy, indicates whether or not to account for parameter drift when simulating forecasts (defaults to TRUE).}
  \item{nrep}{Number of MCMC draws excluding burn-in (defaults to 50000)}
  \item{nburn}{Number of MCMC draws used to initialize the sampler (defaults to 5000). These draws do not enter the computation of posterior moments, forecasts etc.}
   \item{thinfac}{Thinning factor for MCMC output. Defaults to 10, which means that the forecast sequences (\code{fc.mdraws}, \code{fc.vdraws}, \code{fc.ydraws}, see below) contain only every tenth draw of the original sequence. Set \code{thinfac} to one to obtain the full MCMC sequence.}
  \item{itprint}{Print every \code{itprint}-th iteration. Defaults to 10000. Set to very large value to omit printing altogether.}
  \item{save.parameters}{If set to \code{TRUE}, parameter draws are saved in lists (these can be very large). Defaults to \code{TRUE}.}
  \item{k_B, k_A, k_sig, k_Q, k_W, k_S, pQ, pW, pS}{Quantities which enter the prior distributions, see the links below for details. Defaults to the exact values used in the original article by Primiceri.}
}
\value{
  \item{Beta.postmean}{Posterior means of coefficients. This is an array of dimension \eqn{[M, Mp+1, T]}, where \eqn{T} denotes the number of time periods (= number of rows of \code{Y}), and \eqn{M} denotes the number of system variables (= number of columns of \code{Y}). The submatrix \eqn{[, , t]} represents the coefficient matrix at time \eqn{t}. The intercept vector is stacked in the first column; the \code{p} coefficient matrices of dimension \eqn{[M,M]} are placed next to it.}
  \item{H.postmean}{Posterior means of error term covariance matrices. This is an array of dimension \eqn{[M, M, T]}. The submatrix \eqn{[, , t]} represents the covariance matrix at time \eqn{t}.}
  \item{Q.postmean, S.postmean, W.postmean}{Posterior means of various covariance matrices.}
  \item{fc.mdraws}{Draws for the forecast mean vector at various horizons (three-dimensional array, where the first dimension corresponds to system variables, the second to forecast horizons, and the third to MCMC draws). \emph{Note:} The third dimension will be equal to \code{nrep/thinfac}, apart from possible rounding issues.}
  \item{fc.vdraws}{Draws for the forecast covariance matrix. Design similar to \code{fc.mdraws}, except that the first array dimension contains the lower-diagonal elements of the forecast covariance matrix.}
  \item{fc.ydraws}{Simulated future observations. Design analogous to \code{fc.mdraws}.}
  \item{Beta.draws, H.draws}{Matrices of parameter draws, can be used for computing impulse responses later on (see \link{impulse.responses}), and accessed via the helper function \code{\link{parameter.draws}}.
  These outputs are generated only if \code{save.parameters} has been set to \code{TRUE}. }
}
\references{
Del Negro, M. and Primicerio, G.E. (2015). `Time Varying Structural Vector Autoregressions and Monetary Policy: A Corrigendum', Review of Economic Studies 82, 1342-1345. 

Koop, G. and D. Korobilis (2010): `Bayesian Multivariate Time Series Methods for Empirical Macroeconomics', Foundations and Trends in Econometrics 3, 267-358. Accompanying Matlab code available at \url{https://sites.google.com/site/dimitriskorobilis/matlab}.

Primiceri, G.E. (2005): `Time Varying Structural Vector Autoregressions and Monetary Policy', Review of Economic Studies 72, 821-852.
}
\author{
Fabian Krueger, based on Matlab code by Dimitris Korobilis (see Koop and Korobilis, 2010). \emph{Incorporates the corrigendum by Del Negro and Primiceri (2015), which points to an error in the original MCMC algorithm of Primiceri (2005).}
}
\examples{
\dontrun{

# Load US macro data
data(usmacro)

# Estimate trivariate BVAR using default settings
set.seed(5813)
bv <- bvar.sv.tvp(usmacro)

}
}
\seealso{The helper functions \code{\link{predictive.density}} and \code{\link{predictive.draws}} provide simple access to the parameters and forecasts produced by \code{\link{bvar.sv.tvp}}. Impulse responses can be computed using \link{impulse.responses}. For detailed examples and explanations, see the accompanying pdf files hosted at \url{https://github.com/FK83/bvarsv/}.

\emph{Note on priors}: The default values closely follow Primiceri (2005). They have been found to yield reasonable results in several macroeconomic applications, using data sets similar to \code{\link{usmacro}} and \code{\link{usmacro.update}}. However, some of the priors may be implausible in other contexts, especially when the scale of the data is very different from the data sets just mentioned. As is generally the case in Bayesian statistics, the appropriateness of the priors must thus be closely monitored in applications.}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{forecasting methods}
