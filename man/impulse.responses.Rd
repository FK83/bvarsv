\name{impulse.responses}
\alias{impulse.responses}
\title{
Compute Impulse Response Function from a fitted model 
}
\description{
Computes impulse response functions (IRFs) from a model fit produced by \link{bvar.sv.tvp}. The
IRF describes how a variable responds to a shock in another variable, in the periods following the shock. To enable simple handling, this 
function computes IRFs for only one pair of variables that must be specified in advance (see \code{impulse.variable}
and \code{response.variable} below).
}
\usage{
impulse.responses(fit, impulse.variable = 1, response.variable = 2, 
                  t = NULL, nhor = 20, orthogonal = TRUE, draw.plot = TRUE)
}
\arguments{
  \item{fit}{Model fit produced by \link{bvar.sv.tvp}, with the option \code{save.parameters} set to \code{TRUE}.}
  \item{impulse.variable}{Variable which experiences the shock.}
  \item{response.variable}{Variable that (possibly) responds to the shock.}
  \item{t}{Time point from which parameter matrices are to be taken. Defaults to most recent time point.}
  \item{nhor}{Maximal time between impulse and response (defaults to 20).}
  \item{orthogonal}{If \code{TRUE} (the default), impulse responses refer to an orthogonalized shock. If \code{FALSE}, they
  simply refer to a unit shock (see notes below).}
  \item{draw.plot}{If \code{TRUE} (the default): Produces a plot showing the 5, 25, 50, 75 and 95 percent quantiles of the simulated impulse responses.}
}
\value{
List of two elements:
\item{contemporaneous}{Contemporaneous impulse responses (vector of simulation draws).}
\item{irf}{Matrix of simulated impulse responses, where rows represent simulation draws, and columns represent the number of 
  time periods after the shock (1 in first column, \code{nhor} in last column).}
}
\references{
Hamilton, J.D. (1994): Time Series Analysis, Princeton University Press.
}
\author{
Fabian Krueger
}
\examples{
\dontrun{

data(usmacro)
set.seed(5813)
# Run BVAR; save parameters
fit <- bvar.sv.tvp(usmacro, save.parameters = TRUE)
# Impulse responses
impulse.responses(fit)

}
}
\note{
If \code{orthonal} is set to \code{TRUE}, the Cholesky transform (transpose of \link{chol}) is used to produce the orthogonal impulse responses. See Hamilton (1994), Section 11.4, and particularly Equation [11.4.22]. As discussed by Hamilton, the ordering of the system variables matters, and should be considered carefully. The magnitude of the shock (impulse) corresponds to one standard deviation.

If \code{orthonal} is set to \code{FALSE}, the function simply outputs the matrices of the model's moving average representation, see Equation [11.4.1] in 
Hamilton (1994). The scenario considered here may be unrealistic, in that an isolated shock may be unlikely. The magnitude of the shock (impulse)
corresponds to one unit.
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{impulse response analysis}
