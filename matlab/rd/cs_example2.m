%Cleans environment
clear all;
close all;
clc;

%To perform recovery by minimizing l1-norm
addpath(genpath('sparsify_0_4'));
addpath(genpath('cs'));
%length of the signal
N=256;

%Number of random observations to take
M=32;

%Discrete frequency of two sinusoids in the input signal
k1=9;
k2=30;

n=0:N-1;

%Sparse signal in frequency domain.
x=sin(2*pi*(k1/N)*n)+sin(2*pi*(k2/N)*n);



xf=fft(x);




%creating dft matrix
B=dftmtx(N);
Binv=inv(B)*N/sqrt(M);



%creating measurement pattern
% p=randperm(N,M);
p =[ 59,  28, 231,  79, 202, 249, 122,  79, 101,  20,  13, 244,  69,177,  97, 205,  37, 232,  42,  50,  46, 129,  59,  72,   7, 200,43, 117, 145,  40, 142, 171];
p=p+1;
%taking random time measurements
y=x(p)';

%Measurement matrix is built by taking random rows from DFT matrix
A=Binv(p,:);

%Recovery algorithm
% s=greed_omp(y,A,N);

% %%%%%%%%%%%%%%%%%%for AMP%%%%%%%%%%%%%
Eta=@soft_thresholding_C;
Etader=@CalculateSoftThresholdDerivativeComplex;
par=cell(2,1);
par{1}=0; par{2}=50;

[ empiricaliterwatch_sigma, s ] = GenericCAMP( y,A,Eta,Etader,par );
plot(empiricaliterwatch_sigma);

figure;
plot(abs(s));
title('Recovered Spectrum');
grid on;

figure;

subplot(2,1,1);
plot(x)
grid on;
hold on;
stem(p,y,'r');

legend('Original signal','Random samples');
xlabel('Samples');
ylabel('Amplitude');
title('Original Signal,256 samples with two different frequency sinsuoids');

subplot(2,1,2);
plot(abs(xf))
grid on;
xlabel('Samples');
ylabel('Amplitude');
title('Frequency domain, 256 coefficients with 4-non zero coefficients');

% plot(real(B*s)/N,'b');



figure;
grid on;
hold on;
plot(x,'b');
plot(real(B*s)/N,'r');
legend('Original signal','Recovered signal');


function eta=soft_thresholding_C(x,lambda)
% complex soft-thresholding function

eta=(abs(x)> lambda).*(abs(x)-lambda).*(x)./abs(x+eps);

end

function [d1, d2] = CalculateSoftThresholdDerivativeComplex(xc, lambda)

% function [dxR, dxI] = CalculateSoftThresholdDerivativeComplex(xc, lambda)
% This function calculates all the derivatives of the complex soft thresholding
% This is used to predict the correction term for the complex AMP
% algorithm.

xr = real(xc);
xi = imag(xc);
absx3over2 = (xr.^2+xi.^2).^(3/2)+eps;
indicatorabsx = (xr.^2+xi.^2>lambda^2);


dxR(:,1) = indicatorabsx.*(1- lambda*xi.^2./absx3over2);
dxR(:,2) = lambda*indicatorabsx.*xr.*xi./absx3over2;

%
dxI(:,1) = lambda*indicatorabsx.*xr.*xi./absx3over2;
dxI(:,2) =indicatorabsx.*(1- lambda*(xr.^2)./absx3over2); 

d1=dxR(:,1);
d2=dxI(:,2);

end