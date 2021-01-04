function hbeta = ffl_B1xB2xB3(X,y,lambda,LW)
% %3.29.2018
% r is rank, p_r is the dimension of beta_r, so r*beta_r should equal to size(X,2)-1.

%4.2.2018
%LW is a 1-by-r vector, representing weights for different lambda's, at the
%current mode

% This is the setting that one coeffients impose no penalization is allowed. 
% !!!!!!!!!!!! Need to put the coefficients of X that wish to impose no penalization in the last
% column of X.
% For code testing only
% X = randn(100,81);
% y = X * [ones(20,1);-2*ones(10,1);0.5*ones(10,1); 2*ones(20,1);ones(20,1);-3] + randn(size(X,1),1);
% r = 4;
% p_r = 20;
% LW=[0,0,0,0];
% lambda=1;

%% Main Function Code
p=size(X,2)-1;
r=size(LW,2);
p_r=p/r;
n=size(X,1);

D0=diag(-1*ones(p,1));
D=D0(1:(p-1),:)+[zeros(p-1,1),diag(ones(p-1,1))];
rows_to_remove = p_r:p_r:(p-p_r);
D(rows_to_remove,:) = [];

LW2 = repelem(LW,p_r); 
D = D*diag(LW2);
WithPenInd = find(any(D ~= 0,1));
NoUseColumn = find(all(D == 0,1));
NoUseRow = find(all(D == 0,2));
D(NoUseRow,:)=[];
D(:,NoUseColumn)=[];



fill_r = ones(1,p_r);
fill_r_repeat = repmat(fill_r, 1, (size(WithPenInd,2)/p_r));        % Repeat Matrix
fill_r_cell = mat2cell(fill_r_repeat, size(fill_r,1), repmat(size(fill_r,2),1,(size(WithPenInd,2)/p_r))); % Create Cell Array Of Orignal Repeated Matrix
fill = blkdiag(fill_r_cell{:});
D = [D;fill];
%%% Note: the D matrix still need to multiply a lambda vector, so that each
%%% block(each r) have a different penalization.
clear fill;

p_penalize = size(WithPenInd,2);
XtD=X(:,WithPenInd)/D;   %% The first p column of X correponds to the tensor, last column impose no penalization.
X1=XtD(:,1:(p_penalize-(p_penalize/p_r)));
X2=[XtD(:,(p_penalize-(p_penalize/p_r)+1):p_penalize),X(:,NoUseColumn),X(:,(p+1))];
P=(X2/(X2.'*X2))*X2.';
ty=(diag(ones(n,1))-P)*y;
tX=(diag(ones(n,1))-P)*X1;

theta1=lasso(tX,ty,'lambda',lambda,'Standardize',false);

theta2=(X2.'*X2) \ X2.'* (y-X1*theta1);
hbeta0_1=D\[theta1;theta2(1:(p_penalize/p_r))]; %%% hbeta0 is the coefficients for tensor
hbeta0_2=theta2((p_penalize/p_r+1):(p_penalize/p_r+size(NoUseColumn,2)),:);
hbeta0=zeros(p,1);
hbeta0(WithPenInd)=hbeta0_1;
hbeta0(NoUseColumn)=hbeta0_2;
alpha=theta2((p_penalize/p_r+size(NoUseColumn,2)+1),:); %%% alpha is the coefficinets for the last comlumn of X
hbeta=[hbeta0;alpha];
%disp(hbeta); 
end