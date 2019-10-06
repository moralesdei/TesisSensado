function [ empiricaliterwatch_sigma, xhat ] = GenericCAMP( y,A,Eta,Etader, par )

% This algorithm is an extension of AMP and SE to the complex setting
% Inputs:
%   y :          observations
%   A :          a function handle that represents matrix A, A(x,1) means
%                A*x, A(x,2) means A'*x 
%   Eta :        a function handle which is a generic denoiser, 
%                xhat=Eta(temp,sigma) 
%   Etader :     a function handle which is the derivative function of the 
%                denoise function Eta. This function takes a vector and a 
%                value of thresholding (similar as the parameters that the 
%                function Eta needs), and should return two vectors, the 
%                first vector is the partial derivative of real(Eta(temp,sigma))
%                with respect to real(temp), the second vector is the partial
%                derivative of imag(Eta(temp,sigma)) with respect to imag(temp).
%                If you can't provide this derivative
%                function, please input "Null".
%   niter :      the maximum number of iterations
%   par:         a cell with two elements, the first denotes whether we need 
%                all the estimates in whole process of just the final estimate 
%                we obtain, "1" means need, "0" means not; the second
%                denotes how many iteration you want AMP to run, if you
%                input a positive integer t, then AMP runs t times
%                iterations for you, if you input the string 'Auto', then
%                AMP will try to runs 100 times iterations and stop when
%                the ratio of l2 norm of (x_(t+1)-x_(t)) and l2 norm of
%                x_(t) less then 0.01


% Related functions: Eta_der_Estimate_CAMP

% check: @A,@Eta,@Etader, sigma is the estimated std instead of variance.
% sigma_w seems useless in this function

Af  = @(A,x) A*x; % Para 1
At  = @(A,x) A'*x; % Para 2
n=length(y);
lengthN=At(A,zeros(n,1));
N=length(lengthN);

y=y-mean(y);


%%%%% Normalize A

%disp('Normalizing A matrix, please wait...');

pick=randperm(N);
DtmIndex=pick(1:5);
DtmNorms=zeros(5,1);
I=eye(N);
for i=1:5
    DtmNorms(i)=sum(abs( Af(A,I(:,DtmIndex(i))) ).^2);
end

if (sum(DtmNorms>1.1)+sum(DtmNorms<0.9))>0 % We need to normalize A matrix
    disp('It is necessary to normalize the A matrix, please wait...');
    tempA=zeros(n,N);
    colnormA=zeros(N,1);
%     tempA_ra=zeros(n,N);
%     normalize_time_total=0;
    
    

%     for j=1:N                              %remove average
%         tempA_ra(:,j)=tempA(:,j)-mean(tempA(:,j));
%     end
    
    tempA_ra=A-mean(mean(A));
    
    for j=1:N
    colnormA(j)=norm(tempA_ra(:,j));
    end;
    ind=find(colnormA==0);
    colnormA(ind)=(sqrt(sum(abs(tempA(:,ind)).^2,1)))';
    A=tempA_ra;
    disp('Normalizing ends, Iteration starting...');
else
    disp('It is not necessary to normalize the A matrix, Iteration starting...');
    colnormA=ones(N,1);
end

% Denote normalized A matrix into matirx AA, then, when we calculate AA*v, we need 
% to do A*(v./colnormA); when we calculate AA'*v, we need to do (A'*v)./colnormA.

%  A=tempA_ra;
%colnormA=ones(N,1);


par1=logical(par{1});
par2=par{2};
if ischar(par2)
    niter=100;
else
    niter=par2;
end

empiricaliterwatch_sigma=zeros(niter+1,1);
xall=zeros(N,niter+1);

mx=zeros(N,1);
mz=y-Af(A,mx./colnormA);

iteration_time_total=0;
for iter=1:niter
    t0=cputime;
    disp(['iteration = ' num2str(iter)]);
    temp_z=At(A,mz)./colnormA+mx;
     sigma_hat= 1/sqrt(log(2))*median(abs(temp_z));
%    sigma_hat=sqrt(mean(abs(temp_z).^2));
    mx=Eta(temp_z,sigma_hat);
    
    if strcmpi(Etader,'Null')
        mz=y-Af(A,mx./colnormA)+mz*Eta_der_Estimate_CAMP(temp_z,sigma_hat,Eta )*N/n;
    else
        [etaderR,etaderI]=Etader(temp_z,sigma_hat);
        mz=y-Af(A,mx./colnormA)+mz*(sum(etaderR)+sum(etaderI))/(2*n);
    end

    empiricaliterwatch_sigma(iter)=sigma_hat;
    xall(:,iter+1)=mx./colnormA;    
    
    if niter==100 && abs( empiricaliterwatch_sigma(iter)- 1/sqrt(log(2))*median(abs(At(A,mz)./colnormA+mx)) )<0.0001
    %if niter==100 && abs(empiricaliterwatch_sigma(iter)- sqrt(mean(abs(A(mz,2)./colnormA+mx).^2)) )<0.0001
        break;
    end
    
    iteration_time=(cputime-t0)/60;
    iteration_time_total=iteration_time_total+iteration_time;
    iteration_time_remain=iteration_time_total*niter/iter;
    disp(['Iteration #' num2str(iter) ' takes about ' num2str(iteration_time) ' minutes.' 10 'The estimated remaining time for CAMP Algorithm is at most ' num2str(iteration_time_remain) ' minutes.']);

end

empiricaliterwatch_sigma(iter+1)=1/sqrt(log(2))*median(abs(At(A,mz)./colnormA+mx));
%empiricaliterwatch_sigma(iter+1)=sqrt(mean(abs(A(mz,2)./colnormA+mx).^2));

if niter==100
    empiricaliterwatch_sigma=empiricaliterwatch_sigma(1:(iter+1));
    xall=xall(:,1:(iter+1));
end

if iter==100
    fprintf('Iteration reaches the maximum (100) times,\nthe algorithm does not converge within 100 iterations.\n')
end

if par1
    xhat=xall;
else
    xhat=xall(:,end);
end

end