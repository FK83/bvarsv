\name{usmacro}
\alias{usmacro}
\docType{data}
\title{
US macroeconomic time series
}
\description{
Real GDP growth rate, inflation, and treasury bill interest rate for the US (269 quarterly observations from 1947:Q2 to 2014:Q2).
}

\format{
Multiple time series (\code{mts}) object, series names: ``gdp'', ``inf'' and ``tbi''.
}

\source{Federal Reserve Bank of St. Louis (2014): ``Federal reserve economic data'',
\url{http://research.stlouisfed.org/fred2/}.
Accessed: 2014-08-14.

\emph{Note: The underlying FRED series are coded ``GDPC96'', ``GDPDEF'' and ``TB3MS''. For the gdp and inflation series, we take 400 times the log growth rate of the original series (i.e., annualized log growth rates). For the treasury bill interest rate, we compute the quarterly average of the monthly FRED series.}
}

\examples{
\dontrun{

# Load and plot data
data(usmacro)
plot(usmacro)

}
}



\keyword{datasets}