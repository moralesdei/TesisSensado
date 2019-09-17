%% Random Demodulator Recovery Algorithm by Using Simulated Analog Data from LTSpice.
%
%Before using this script run rd_demo_ltsipce_gen script, run LTspice
%simulation and export vout.
%
%Alexander López-Parrado (2017)


%Cleans environment
clear all;
close all;
clc;

%To perform recovery by minimizing l1-norm
% addpath(genpath('sparsify_0_4'));
addpath(genpath('cs'));

%Loads environment (saved rd_Demo_ltspice_gen_signals)
load('environment');




% there are L Nyquist periods in one period of p(t) (for Tropp's analysis
% Tx=Tp or L=N)

L=N;

%%%%%%%%%%%%%%%%%Model of analog low-pass filter, should match LTSpice RC
%%%%%%%%%%%%%%%%%filter  %%%%%%%%%%%%%%%%%%%%%%%%

%1-st order RC low-pass filter
tau=1/(2*pi*fc);
B=1;
A=[tau 1];


%Transfer function of analog low-pass filter
lpf=tf(B,A);

%Estimation of impulse response for analog filter by making bilinear
%transform
lpf_d=c2d(lpf,1/(W),'tustin');
[B_d,A_d]=tfdata(lpf_d,'v');
h=impz(B_d,A_d,N/M);




%Reads LTSpice simulation results
if(is_gilbert_cell) 
    LTSpicedata=importdata('../../ltspice/test-hfa3101-rc-lpf.txt','\t',1);
    %Gilbert-cell has gain
    MixerGain=8;
    
else
    LTSpicedata=importdata('../../ltspice/test-adg609-rc-lpf.txt','\t',1);
    %Analog muliplexer has no gain
    MixerGain=1.0;
end

tspice=LTSpicedata.data(:,1);
lpfoutputI=LTSpicedata.data(:,3);%-mean(LTSpicedata.data(:,2));
lpfoutputQ=LTSpicedata.data(:,2);%-mean(LTSpicedata.data(:,3));

   lpfoutput=(lpfoutputI+1i*lpfoutputQ);

%Finds closest time value to perform sampling
    [dt_start ,i_tstart]=(min(abs(tspice-Td)));
 
    
  lpfoutput=lpfoutput(i_tstart:end);
  tspice=tspice(i_tstart:end);



 lpfoutput=lpfoutput-mean(lpfoutput);
 
 
lpfoutput=lpfoutput/MixerGain;



%Time to start sampling after dummy training signal
tstart=1/fsn-1/(2*W)+Td;

%ADC simulation
z=zeros(1,M);
for k=1:M
    %Finds closest time value to perform sampling
    [dt_start ,i_tstart]=(min(abs(tspice-tstart)));
    %Sampling
    z(k)=lpfoutput(i_tstart);
    %Next sampling instant
    tstart=tstart+1/fsn;
end;


%Recovery algorithm from LTSpice samples
[x_hat,X_hat,freq_hat]=rd_recovery2(z,W,S,h,N,M,Tx); % recover input signal

%% Results
% fprintf('Nyquist rate %3.3f Hz\n',W);
% fprintf('System sampling rate %3.3f Hz\n',(M*W)/N);
% fprintf('Subsampling factor %3.3f\n',M/N);
% fprintf('Original Frequencies (Hz)\n');
% fprintf('%6.10f\n',sort(-freq/Tx));
% fprintf('Recovered Frequencies (Hz)\n');
% fprintf('%6.2f\n',freq_hat/Tx);
% fprintf('Original FS Coefficients (real)\n');
% fprintf('%6.2f\n',flipud(X(X>0)));
% fprintf('Recovered FS Coefficients (real)\n');
% fprintf('%6.2f\n',X_hat(X_hat>0));
% fprintf('Average squared error =%2.10f\n',sum((1/length(x))*(real(x)-real(x_hat)).^2));

figure % graphically display recovered frequencies and FS coefficients compared to true values
stem(-N/2:N/2-1,fftshift(X)); hold on;
stem(-N/2:N/2-1,abs((X_hat)),'r','Marker','*');
xlabel('Frequency Index'); ylabel('Magnitude'); title('FS Coefficients (Magnitude)');
legend('Original FS coefficients','Recovered FS coefficients')

figure % plot fft of original multitone signal x and reconstructed signal x_hat
f= (-length(x)/2:length(x)/2-1)./length(x) * (W);
%h1=subplot(211); 
plot(f/1e6,abs(fftshift(fft(x))),'Marker','none');
grid on;
title('Fourier Spectrum--Input signal'); xlabel('MHz'); ylabel('Magnitude')
%h2=subplot(212); 
figure;
plot(f/1e6,abs(fftshift(fft(x_hat))),'Marker','none');
grid on;
title('Fourier Spectrum--Reconstructed signal');xlabel('MHz'); ylabel('Magnitude')
% linkaxes([h1 h2])

figure % plot original multitone signal x and reconstructed signal x_hat
h3=subplot(311); plot(t/1e-6,real(x),'Marker','none');
xlabel('\mu s'); title('Original Signal (real part)');
h4=subplot(312); plot(t/1e-6,real(x_hat),'r','Marker','none');
xlabel('\mu s'); title('Reconstructed Signal (real part)');
h5=subplot(313); plot(t/1e-6,real(x)-real(x_hat),'Marker','none');
xlabel('\mu s'); title('Difference Signal');
linkaxes([h3 h4 h5])

