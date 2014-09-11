#include "RcppArmadillo.h"
#include <cmath>
#define _USE_MATH_DEFINES
#include <math.h>

// Rcpp::depends(RcppArmadillo)

using namespace Rcpp;        

// [[Rcpp::export]]
arma::colvec mvndrawC(arma::colvec mu, arma::mat sig) {

double k = mu.size();
arma::colvec aux = as<arma::colvec>(rnorm(k));
arma::mat csig = arma::chol(sig).t();
arma::colvec out = mu + csig*aux;
return(out);

}
 
 // [[Rcpp::export]]
List carterkohn(arma::mat y, arma::mat Z, arma::mat Ht, arma::mat Qt, double m, double p, double t, arma::colvec B0, arma::mat V0) {
 
arma::colvec bp = B0; // reuses memory and avoids extra copy
arma::mat Vp = V0;
arma::colvec btt = B0;	// initialize now s/t that their scope extends beyond loop
arma::mat Vtt = V0;
arma::mat f = arma::zeros(p, Ht.n_cols);
arma::mat invf = f;
arma::colvec cfe = arma::zeros(y.n_rows);
arma::mat bt = arma::zeros(t,m);
arma::mat Vt = arma::zeros(pow(m,2),t);
arma::colvec loglik = arma::zeros(1);
arma::mat R = arma::zeros(p,p);
arma::mat H = arma::zeros(p,m);
 
for (int i = 1; i < (t+1); i++) {
  R = Ht.rows((i-1)*p,i*p-1);
  H = Z.rows((i-1)*p,i*p-1);

  cfe = y.col(i-1) - H*bp;   // conditional forecast error
  f = H*Vp*H.t() + R;    	  // variance of the conditional forecast error
  invf = f.i();		  // invert only once
  loglik = loglik + log(det(f)) + cfe.t()*invf*cfe;

  btt = bp + Vp*H.t()*invf*cfe;
  Vtt = Vp - Vp*H.t()*invf*H*Vp;

  if (i < t){
    bp = btt;
    Vp = Vtt + Qt;
  }

  bt.row(i-1) = btt.t();
  Vt.col(i-1) = arma::vectorise(Vtt);
}

arma::mat bdraw = arma::zeros(t,m);
bdraw.row(t-1) = mvndrawC(btt,Vtt).t();   

for (int i = 1; i < t; i++) {	// Backward recursions
    arma::colvec bf = bdraw.row(t-i).t();
    btt = bt.row(t-i-1).t();
    Vtt = arma::reshape(Vt.col(t-i-1),m,m);
    f = Vtt + Qt;
    invf = f.i();
    cfe = bf - btt;
    arma::colvec bmean = btt + Vtt*invf*cfe;
    arma::mat bvar = Vtt - Vtt*invf*Vtt;
    bdraw.row(t-i-1) = mvndrawC(bmean, bvar).t();
}

return List::create(Named("loglik") = loglik, Named("bdraws") = bdraw.t());
}

// [[Rcpp::export]]
arma::mat alphahelper(arma::mat y, arma::mat Z, arma::mat Btdraw){

double M = y.n_rows;
double t = y.n_cols;
arma::mat yhat = arma::zeros(M,t);

for (int i = 1; i < (t+1); i++) {
  yhat.col(i-1) = y.col(i-1) - Z.rows((i-1)*M,i*M-1)*Btdraw.col(i-1);
}

return yhat;

}

// [[Rcpp::export]]
arma::mat sigmahelper1(arma::mat Atdraw, double M){
  
double t = Atdraw.n_cols; 
  
arma::mat capAt = arma::zeros(M*t,M);
for (int i = 1; i < (t+1); i++) {
  arma::mat capatemp(M,M);
  capatemp.eye();
  arma::colvec aatemp = Atdraw.col(i-1);
  double ic = 1;
  for (int j = 2; j < (M+1); j++) {
    capatemp(arma::span(j-1,j-1),arma::span(0,j-2)) = aatemp.rows(ic-1,ic+j-3).t();
    ic = ic + j - 1;
  }
  capAt.rows((i-1)*M,(i*M)-1) = capatemp;
}

return capAt;
}

// [[Rcpp::export]]
List sigmahelper2(arma::mat capAt, arma::mat yhat, arma::colvec qs, arma::colvec ms, arma::colvec u2s, arma::mat Sigtdraw, arma::mat Zs, arma::mat Wdraw, arma::colvec sigma_prmean, arma::mat sigma_prvar){

double M = capAt.n_cols;
double t = capAt.n_rows / M; 

arma::mat y2 = arma::zeros(M,t);
for (int i = 1; i < (t+1); i++) {
  y2.col(i-1) = pow(capAt.rows((i-1)*M,(i*M)-1) * yhat.col(i-1),2);
}

arma::mat aux = 0.001 * arma::ones(t,M);
arma::mat yss = log( aux + y2.t() );

arma::colvec cprw = arma::zeros(7,1);
arma::mat statedraw = arma::zeros(t,M);
for (int jj = 1; jj < (M+1); jj++) {
  for (int i = 1; i < (t+1); i++) {
  arma::colvec prw = arma::zeros(7,1);
    for (int k = 1; k < 8; k++) {
      prw(k-1) = qs(k-1) * (1/sqrt(2*M_PI*u2s(k-1)))*exp(-0.5*((pow(yss(i-1,jj-1) - Sigtdraw(jj-1,i-1) - ms(k-1) + 1.2704,2))/u2s(k-1)));
    }
  cprw = arma::cumsum(prw/arma::sum(prw));
  double trand = as<double>(runif(1));
  double imix = 0;
  if (trand < cprw[0]){
    imix = 1;
  } else if (trand < cprw[1]) {
    imix = 2;
  } else if (trand < cprw[2]) {
    imix = 3;
  } else if (trand < cprw[3]) {
    imix = 4;
  } else if (trand < cprw[4]) {
    imix = 5;
  } else if (trand < cprw[5]) {
    imix = 6;
  } else if (trand < cprw[6]) {
    imix = 7;
  }
  statedraw(i-1,jj-1) = imix;  
  }
}

arma::mat vart = arma::zeros(t*M,M);
arma::mat yss1 = arma::zeros(t,M);
for (int i = 1; i < (t+1); i++) {
  for (int j = 1; j < (M+1); j++) {
    double imix = statedraw(i-1,j-1);
    vart(((i-1)*M+j-1),j-1) = u2s(imix-1);
    yss1(i-1,j-1) = yss(i-1,j-1) - ms(imix-1) + 1.2704;
  }
}

arma::mat Sigtdraw_new = carterkohn(yss1.t(),Zs,vart,Wdraw,M,M,t,sigma_prmean,sigma_prvar)["bdraws"];

arma::mat sigt = arma::zeros(M*t,M);
for (int i = 1; i < (t+1); i++) {
  arma::mat sigtemp = arma::zeros(M,M);
  sigtemp.diag() = exp(0.5*Sigtdraw_new.col(i-1));
  sigt.rows((i-1)*M,(i*M)-1) = sigtemp;
}

return List::create(Named("Sigtdraw") = Sigtdraw_new, Named("sigt") = sigt);
}

// [[Rcpp::export]]
List sigmahelper3(arma::mat capAt, arma::mat sigt){

double M = sigt.n_cols;
double t = sigt.n_rows/M;
arma::mat Ht = arma::zeros(M*t,M);
arma::mat Htsd = arma::zeros(M*t,M);

for (int i = 1; i < (t+1); i++) {
  arma::mat inva = capAt.rows((i-1)*M, (i*M)-1).i();
  arma::mat stem = sigt.rows((i-1)*M, (i*M)-1);
  arma::mat Hsd = inva*stem;
  Ht.rows((i-1)*M, (i*M)-1) = Hsd * Hsd.t();
  Htsd.rows((i-1)*M, (i*M)-1) = Hsd;
}

return List::create(Named("Ht") = Ht, Named("Htsd") = Htsd);
}

arma::mat corrvc(arma::mat x){

arma::mat aux = sqrt(x.diag());
return (x / (aux * aux.t()));

}

// [[Rcpp::export]]
List getvc(arma::mat Ht){
  
double M = Ht.n_cols;
double t = Ht.n_rows / M;

arma::mat dat1 = arma::zeros(t,M);
arma::mat dat2 = arma::zeros(t,M*(M-1)*0.5);
  
for (int i = 1; i < (t+1); i++) {
  
  arma::mat aux = arma::zeros(1,M);
  
  for (int j = 1; j < (M+1); j++) { 
    aux(0,j-1) = sqrt(Ht((i-1)*M+j-1,j-1));
  }
  
  dat1.row(i-1) = aux;
  
  arma::mat aux2 = corrvc( Ht.rows((i-1)*M, (i*M)-1) );
  arma::mat aux3 = arma::zeros(1,1);
  aux3(0) = aux2(1,0);
  
  if (M > 2){
    double ic = 2;
    for (int j = 3; j < (M+1); j++) { 
      arma::mat aux4 = aux2(arma::span(j-1,j-1), arma::span(0, ic-1));
      aux3 = arma::join_rows(aux3, aux4);
      ic = ic + 1;
    }    
  }
  
  dat2.row(i-1) = aux3;
 
} 
 
return List::create(Named("out1") = dat1, Named("out2") = dat2); 
 
} 

// [[Rcpp::export]]
arma::mat makeregs_fcC(arma::mat ydat, double p){

double M = ydat.n_cols;
arma::mat out = arma::zeros(M,M);
out.eye();
arma::mat aux = out;

for (int i = 1; i < (p+1); i++) {
  
  arma::mat tmp = ydat.row(ydat.n_rows-i);
  out = arma::join_rows(out, arma::kron(aux, tmp));
  
}
  
return(out);
}   

// [[Rcpp::export]]
arma::colvec mz(double n){
  arma::colvec aux = arma::zeros(n);
  return(aux);
}

// [[Rcpp::export]]
arma::mat meye(double n){
  arma::mat aux = arma::zeros(n,n);
  aux.eye();
  return(aux);  
}

// [[Rcpp::export]]
arma::colvec vechC(arma::mat x) {

double n = x.n_rows;
arma::colvec out = arma::zeros(0.5*n*(n+1));
double ct = 0;

for (int ii = 1; ii < (n+1); ii++) { 

  out.rows(ct, ct+n-ii) = x(arma::span(ii-1,n-1), arma::span(ii-1, ii-1));
  ct = ct + n + 1 - ii;

}

return(out);

}

// [[Rcpp::export]]
List getfcsts(arma::colvec Bt0, arma::colvec At0, arma::colvec Sigt0, arma::mat Qdraw, arma::mat Sdraw, arma::mat Wdraw, arma::mat ydat, double nf, double p){
  
// Initialize parameters
arma::colvec Btfc = Bt0;
arma::colvec Atfc = At0;
arma::colvec Sigtfc = Sigt0;
arma::mat ystar = ydat;
double M = ystar.n_cols;
arma::mat fcd = arma::zeros(M,nf);
arma::mat fcm = arma::zeros(M, nf);
arma::mat fcv = arma::zeros(M*(M+1)*0.5, nf);

for (int hh = 1; hh < (nf+1); hh++) { 

  // Draw Bt
  Btfc = Btfc + mvndrawC(mz(Btfc.n_rows),Qdraw);
  // Draw At
  Atfc = Atfc + mvndrawC(mz(Atfc.n_rows),Sdraw);
  // Draw Sigt
  Sigtfc = Sigtfc + mvndrawC(mz(Sigtfc.n_rows),Wdraw);
  
  // Create the VAR covariance matrix
  arma::mat aux1 = Atfc;
  arma::mat capAtfc = sigmahelper1(aux1,M);
  arma::mat aux2 = arma::zeros(M,M);
  aux2.diag() = exp(0.5*Sigtfc);
  arma::mat Hfc = capAtfc.i() * aux2;
  Hfc = Hfc * Hfc.t();
  
  // save fc variance
  fcv.col(hh-1) = vechC(Hfc); 
   
  // make & save fc mean
  arma::mat Zfc = makeregs_fcC(ystar,p);
  arma::colvec mtemp = Zfc*Btfc;
  fcm.col(hh-1) = mtemp;
  
  // draw & save realzation
  arma::colvec ytemp = mvndrawC(mtemp, Hfc);
  ystar = arma::join_cols(ystar,ytemp.t());
  fcd.col(hh-1) = ytemp;
  
}

return List::create(Named("mean") = fcm, Named("variance") = fcv, Named("draw") = fcd); 
 
}

// [[Rcpp::export]]
arma::mat wishdrawC(arma::mat h, double n) {

double k = h.n_rows;
arma::mat out = arma::zeros(k,n);

for (int i = 1; i < (n + 1); i++) {
  out.col(i-1) = mvndrawC(mz(k),h);
}

out = out * out.t();
return(out);

}

// [[Rcpp::export]]
arma::rowvec makeregs2_fcC(arma::mat dat, double p) {

double M = dat.n_cols;
double t = dat.n_rows;
arma::rowvec out(p*M+1);
out.fill(1);
for (int ii = 1; ii < (p+1); ii++) { 
  out.cols((ii-1)*M+1,ii*M) = dat.row(t-ii);
}
return(out);

} 

// [[Rcpp::export]]
arma::mat matmult(arma::mat x, double nt){
arma::mat out =  meye(x.n_rows);
if (nt == 1){
  out = x;
} else if (nt > 1) {
  arma::mat tempmat = x;
    for (int ii = 1; ii < (nt); ii++) {
      tempmat = tempmat * x;
    }
  out = tempmat;
} 
return(out);
}

// [[Rcpp::export]]
List varfcst(arma::mat b, arma::mat sig, arma::mat y, double nf){
double k = b.n_rows;
double p = (b.n_cols - 1)/k;
arma::colvec bigy = arma::zeros(p*k);
// Define y vector in companion form
for (int ii = 1; ii < (p+1); ii++) {
  bigy.rows((ii-1)*k,ii*k-1) = y.row(p-ii).t();
}
arma::colvec nu = b.col(0);   // intercept vector 
arma::mat a = b.cols(1,b.n_cols-1); // VAR coefficient matrices
if (p > 1){
  arma::colvec tmp = arma::zeros(k*(p-1));
  nu = arma::join_cols(nu,tmp);
  arma::mat tmp2 = meye(k*(p-1));
  arma::mat tmp3 = arma::zeros(k*(p-1),k);
  a = arma::join_cols(a, arma::join_rows(tmp2, tmp3));  
}
arma::mat om0 = sig;
if (p > 1){
  arma::mat tmp4 = arma::zeros(k, k*(p-1));
  arma::mat tmp5 = arma::zeros(k*(p-1), k*p);
  om0 = arma::join_cols(arma::join_rows(sig, tmp4), tmp5);  
}
arma::mat fcv = om0;
arma::mat aux = meye(k*p);
for (int hh = 1; hh < (nf); hh++) {
  aux = aux + matmult(a, hh);
  fcv = fcv + matmult(a, hh)*om0*(matmult(a, hh).t());
}
arma::mat aux2 = aux*nu + matmult(a, nf)*bigy;
return List::create(Named("mean") = aux2.rows(0, k-1), Named("variance") = fcv(arma::span(0,k-1), arma::span(0, k-1))); 
}   