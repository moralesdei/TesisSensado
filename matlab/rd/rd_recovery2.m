function [x_hat,X_hat,freq_hat]=rd_recovery2(y,W,S,h,N,M,Tx)
% This script accepts as input the compressed samples y and returns estimates (if not the 
% exact values) of the original signal x (specifically returns estimates of frequency support,
% amplitudes of Fouries series coefficients, and amplitudes of time domain signal).

% Usage: [x_hat,X_hat,freq_hat]=rd_recovery(y,t,S,N,M,Tx)
% y: sampled output (for this simulation, y is a sub-sampled version of x)
% t: time interval (vector) over which signal is observed (sec)
% S: vector of random +/- 1's 
% h: Impulse response of filter
% N: number of Nyquist periods within [0,Tx]
% M: the ratio M/N specifies rate of sampling process relative to Nyquist rate 
% Tx: length of observation interval 
% x_hat: recovered (estimated) spectrally sparse multitone signal
% X_hat: recovered (estimated) FS coefficients
% freq_hat: recovered frequencies (support)

% Copyright (c) 2010 by Michael A Lexa, Mike E Davies, John S Thompson
% Institute of Digital Communications, University of Edinburgh
%
% This work is licenced under the Creative Commons Attribution 3.0
% Unported License. To view a copy of this licence, visit
% http://creativecommons.org/licenses/by/3.0/ or send a letter to
% Creative Commons, 171 Second Street, Suite 300, San Francisco,
% California 94105, USA.

% create matrices characterizing the RD samples/measurements; see the 
% technical report "Sampling Sparse Multitone Signals with a Random Demodulator" 
% included with the CTSS Sampling Toolbox
l=0:N-1;   n=-N/2:N/2-1;
Psi=exp(1i*(2*pi/N)*l'*n); 
D=diag(S);
H=zeros(M,N);
%Reflects impulse response
 H(1,1:N/M)=flipud(h(:));
for i=2:M
 H(i,:)=circshift(H(i-1,:),[0 N/M]);
end
A=H*D*Psi; % in accompanying technical report, H*D=Phi
 
% s=greed_omp(y,A,N,'stopTol',N,'verbose',true); % support and FS coeffiecient recovery (T. Blumensath's OMP algorithm)  

%%%%%%%%%%%%%%%%%%%for CoSaMP%%%%%%%%%%%%%
opts            = [];
opts.maxiter    = 50;
% opts.tol        = 1e-8;
% opts.HSS        = true;
% opts.two_solves = true; % this can help, but no longer always works "perfectly" on noiseless data
opts.printEvery = 10;
% K_target    = round(length(b)/3)-1; opts.normTol = 2.0;
K_target        = 50;   % When extremely noisy, this is best; when no noise, this is sub-optimal
% opts.addK       = 2*K_target; % default
% opts.addK       = K_target; % this seems to work a bit better
% opts.addK       = 5;    % make this smaller and CoSaMP behaves more like OMP 
                        % (and does better for the correlated measurement matrix)
 s = OMP( A, y.', K_target, [], opts);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5

index=sort(find(abs(s)>0),'ascend');
X_hat=N*s; 

% Convert indices into corresponding frequencies (Hz)
tones=(-N/2:N/2-1)';
freq_hat=tones(index);

t=0:1/(W):Tx-1/(W); % observation interval in sec [0,Tx]

% Reconstruct Nyquist-rate samples
x_hat=(1/N)*sum(diag(X_hat(index))  *exp(1i*(2*pi)/Tx*-freq_hat*t),1)';