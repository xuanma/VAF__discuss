%% First let's use the data samples provided by MATLAB
clear clc
load hald
X = ingredients;
%% Now using the built in pca function to perform PCA
%  In fact, using cov() and eigs() or svd() can also work, leading to the
%  same results.
k = 2;
[n_samples, n_variables] = size(X);
[coeff,score,latent,tsquared,explained,mu] = pca(X);
% coeff: principal component coefficients
% score: principal component scores
% latent: eigenvalues
% explained: cumsum 
% mu: the mean value of each variable
tranMatrix = coeff(:,1:k);
dr_X = score(:,1:k);
reconstructed_X = dr_X * tranMatrix' + repmat(mu,size(n_samples,1));% Notice the 'mu' here
% Now we have y = X, y_hat = reconstructed_X
ratio = cumsum(latent)./sum(latent);
vaf_from_eigen = ratio(k);
fprintf('The VAF calculated from the cumsum of eigenvalues is %f \n', vaf_from_eigen);
vaf_from_func = calculateVAF(X, reconstructed_X);
fprintf('The VAF calculated from function is %f \n', vaf_from_func);
vaf_from_trace = calculateVAFfromTrace(X, reconstructed_X);
fprintf('The VAF calculated from trace is %f \n', vaf_from_trace);
vaf_by_dim = calculateVAFbyDim(X, reconstructed_X);
fprintf('The VAF calculated dim by dim and then averaged is %f \n', mean(vaf_by_dim));
%%
function VAF = calculateVAF(y, y_hat)
    VAF = 1 - sum(sum( (y_hat - y).^2 )) / sum(sum( (y - repmat(mean(y),size(y,1),1)).^2 ));
end
% Here only the y in denominator is zero-meaned

function VAF = calculateVAFfromTrace(y, y_hat)
    VAF = trace(cov(y_hat))/trace(cov(y));
end
% For this method, in fact the variance of each variable can be obtained
% from diag(cov(X)), then ...

function VAF = calculateVAFbyDim(Y, Yhat)
    %calculate sum of square errors
    SSE = sum((Y-Yhat).^2,1);
    
    %calculate mean of actual output
    meanY = mean(Y,1);
    
    %find sum of squared deviations from mean
    SS = sum((Y-repmat(meanY,size(Y,1),1)).^2);
    
    %find VAF
    VAF = 1-SSE./SS;
end

