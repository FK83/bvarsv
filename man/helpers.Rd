\name{helpers}
\alias{predictive.density}
\alias{predictive.draws}
\alias{parameter.draws}
\title{
Helper Functions to Access BVAR Forecast Distributions and Parameter Draws
}
\description{
Functions to extract a univariate posterior predictive distribution from a model fit generated by \code{\link{bvar.sv.tvp}}.
}
\usage{
predictive.density(fit, v = 1, h = 1, cdf = FALSE)
predictive.draws(fit, v = 1, h = 1)
parameter.draws(fit, type = "lag1", row = 1, col = 1)
}
\arguments{
  \item{fit}{List, model fit generated by \code{\link{bvar.sv.tvp}}}
  \item{v}{Index for variable of interest. \emph{Must be in line with the specification of \code{fit}}.}
  \item{h}{Index for forecast horizon of interest. \emph{Must be in line with the specification of \code{fit}}.}
  \item{cdf}{Set to TRUE to return cumulative distribution function, set to FALSE to return probability density function}
  \item{type}{Character string, used to specify output for function \code{\link{parameter.draws}}. Setting to \code{"intercept"} returns parameter draws for the intercept vector. Setting to one of \code{"lag1"}, ..., \code{"lagX"}, (where X is the lag order used in \code{fit})
  returns parameter draws from the autoregressive coefficient matrices. Setting to \code{"vcv"} returns draws for the elements of the residual variance-covariance matrix.}
  \item{row, col}{Row and column index for the parameter for which \code{\link{parameter.draws}} should return posterior draws. That is, the function returns the row, col element of the matrix specified by \code{type}. Note that 
  \code{col} is irrelevant if \code{type = "intercept"} has been chosen.} 
}
\value{
 \code{\link{predictive.density}} returns a function \code{f(z)}, which yields the value(s) of the predictive density at point(s) \code{z}. This function exploits conditional normality of the model, given the posterior draws of the parameters.
 
\code{\link{predictive.draws}} returns a list containing vectors of MCMC draws, more specifically:
\item{y}{Draws from the predictand itself}
\item{m}{Mean of the normal distribution for the predictand in each draw}
\item{v}{Variance of the normal distribution for the predictand in each draw}
 
Both outputs should be closely in line with each other (apart from a small amount of sampling noise), see the link below for details.

\code{\link{parameter.draws}} returns posterior draws for a single (scalar) parameter of the model fitted by \code{\link{bvar.sv.tvp}}. The output is a matrix, with rows representing MCMC draws, and columns representing time.
}
\author{
Fabian Krueger
}
\examples{
\dontrun{

# Load US macro data
data(usmacro)

# Estimate trivariate BVAR using default settings
set.seed(5813)
bv <- bvar.sv.tvp(usmacro)

# Construct predictive density function for the second variable (inflation), one period ahead
f <- predictive.density(bv, v = 2, h = 1)

# Plot the density for a grid of values
grid <- seq(-2, 5, by = 0.05)
plot(x = grid, y = f(grid), type = "l")

# Cross-check: Extract MCMC sample for the same variable and horizon
smp <- predictive.draws(bv, v = 2, h = 1)

# Add density estimate to plot
lines(density(smp), col = "green")

}
}
\seealso{For examples and background, see the accompanying pdf files hosted at \url{https://github.com/FK83/bvarsv/}.}

\keyword{helpers}