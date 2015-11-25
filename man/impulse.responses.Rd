\name{impulse.responses}
\alias{impulse.responses}
\title{
Compute Impulse Response Function from a Fitted Model 
}
\description{
Computes impulse response functions (IRFs) from a model fit produced by \link{bvar.sv.tvp}. The
IRF describes how a variable responds to a shock in another variable, in the periods following the shock. To enable simple handling, this 
function computes IRFs for only one pair of variables that must be specified in advance (see \code{impulse.variable}
and \code{response.variable} below).
}
\usage{
impulse.responses(fit, impulse.variable = 1, response.variable = 2, 
                  t = NULL, nhor = 20, scenario = 2, draw.plot = TRUE)
}
\arguments{
  \item{fit}{Model fit produced by \link{bvar.sv.tvp}, with the option \code{save.parameters} set to \code{TRUE}.}
  \item{impulse.variable}{Variable which experiences the shock.}
  \item{response.variable}{Variable which (possibly) responds to the shock.}
  \item{t}{Time point from which parameter matrices are to be taken. Defaults to most recent time point.}
  \item{nhor}{Maximal time between impulse and response (defaults to 20).}
  \item{scenario}{If \code{1}, there is no orthogonalizaton, and the shock size corresponds to one unit of the impulse variable. If \code{scenario} is either \code{2} (the default) or \code{3}, 
  the error term variance-covariance matrix is orthogonalized via Cholesky decomposition. For \code{scenario = 2}, the Cholesky decomposition of the error term VCV matrix at time point \code{t} is used. 
  \code{scenario = 3} is the variant used in Del Negro and Primiceri (2015). Here, the diagonal elements are set to their averages over time, whereas the off-diagonal elements are specific to time \code{t}. See the 
  notes below for further information.}
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

Del Negro, M. and Primicerio, G.E. (2015). `Time Varying Structural Vector Autoregressions and Monetary Policy: A Corrigendum', Review of Economic Studies 82, 1342-1345. Supplementary material available at 
\url{http://restud.oxfordjournals.org/content/82/4/1342/suppl/DC1} (accessed: 2015-11-17).
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
If \code{scenario} is set to either \code{2} or \code{3}, the Cholesky transform (transpose of \link{chol}) is used to produce the orthogonal impulse responses. See Hamilton (1994), Section 11.4, and particularly Equation [11.4.22]. As discussed by Hamilton, the ordering of the system variables matters, and should be considered carefully. The magnitude of the shock (impulse) corresponds to one standard deviation of the error term.

If \code{scenario = 1}, the function simply outputs the matrices of the model's moving average representation, see Equation [11.4.1] in 
Hamilton (1994). The scenario considered here may be unrealistic, in that an isolated shock may be unlikely. The magnitude of the shock (impulse)
corresponds to one unit of the error term.

<<<<<<< HEAD
Further supporting information is available at \url{https://sites.google.com/site/FK83research/code}.
=======
Further supporting documents are available at \url{https://sites.google.com/site/FK83research/code}.
>>>>>>> 50f3118e19e2c340d481836c626ee70a54b5574d
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{impulse response analysis}
