\name{Example data sets}
\alias{usmacro}
\alias{usmacro.update}
\docType{data}
\title{
US Macroeconomic Time Series
}
\description{
Inflation rate, unemployment rate and treasury bill interest rate for the US, as used by Primiceri (2005). Whereas \code{usmacro} covers the time period studied by
Primiceri (1953:Q1 to 2001:Q3), \code{usmacro.update} updates the data until 2015:Q2.
}

\format{
Multiple time series (\code{mts}) object, series names: `inf', `une', and `tbi'.
}

\source{
Inflation data provided by Federal Reserve Bank of Philadelphia (2015): `Real-Time Data Research Center', 
\url{https://www.phil.frb.org/research-and-data/real-time-center/real-time-data/data-files/p}
Accessed: 2015-10-29.
The inflation rate is the year-over-year log growth rate of the GDP price index. We use the 2001:Q4 vintage of the price index for \code{usmacro}, and the 2015:Q3 vintage for \code{usmacro.update}. 

Unemployment and Treasury Bill: 
Federal Reserve Bank of St. Louis (2015): `Federal Reserve Economic Data',
\url{http://research.stlouisfed.org/fred2/}.
Accessed: 2015-10-29.
The two series have the identifiers `UNRATE' and `TB3MS'. For each quarter, we compute simple averages over three monthly observations. 
}

\examples{
\dontrun{

# Load and plot data
data(usmacro)
plot(usmacro)

}
}

\references{
Primiceri, G.E. (2005): `Time Varying Structural Vector Autoregressions and Monetary Policy', Review of Economic Studies 72, 821-852.
}



\keyword{datasets}