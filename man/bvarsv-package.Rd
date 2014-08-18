\name{bvarsv-package}
\alias{bvarsv-package}
\alias{bvarsv}
\docType{package}
\title{
Bayesian Analysis of a Vector Autoregressive Model with Stochastic Volatility and Time-Varying Parameters
}
\description{
R/C++ implementation of the Primiceri (2005) model, which allows for both stochastic volatility and time-varying regression parameters. This is a very flexible framework which nests several other models as special cases. The package contains functions for computing posterior predictive distributions from the model, based on an input data set. 
}
\details{
\tabular{ll}{
Package: \tab bvarsv\cr
Type: \tab Package\cr
Version: \tab 1.0\cr
Date: \tab 2014-08-14\cr
License: \tab GPL (>= 2)\cr
URL: https://sites.google.com/site/fk83research/code, https://github.com/FK83/bvarsv
}

}

\author{
Fabian Krueger <Fabian.Krueger83@gmail.com>
Based on Matlab code by Dimitris Korobilis (see Koop and Korobilis, 2010) 
}
\references{
The code incorporates the recent corrigendum by Del Negro and Primiceri (2014), which points to an error in the original MCMC algorithm.

Del Negro, M. and G. Primiceri (2014): ``Time Varying Structural Vector Autoregressions and Monetary Policy: A Corrigendum'', working paper, Northwestern University. 

Koop, G. and D. Korobilis (2010): ``Bayesian Multivariate Time Series Methods for Empirical Macroeconomics'', Foundations and Trends in Econometrics 3, 267-358.

Primiceri, G. (2005): ``Time Varying Structural Vector Autoregressions and Monetary Policy'', Review of Economic Studies 72, 821-852.
}
\keyword{ package }

\examples{
\dontrun{

# Load US macro data
data(usmacro)

# Estimate trivariate model using Primiceri's prior choices (default settings)
bv <- bvar.sv.tvp(usmacro)

}
}
