%7.26.2019
%The proposed method: SHIV

%This code compares the proposed method with voxelwise regression, 
%Zhou's tensor regression, and its variation with lasso penalty

% cd('~/Code') %Change the default directory to the folder "Code"
addpath 'Core subfunctions/NIfTI_Matlab' %nii data 
addpath 'Core subfunctions/tensor_toolbox_2.6'
addpath 'Core subfunctions/TensorReg/code'
addpath 'Core subfunctions/SparseReg/code'
addpath 'Core subfunctions'

% disp('addpath complete');

clear;

% seed = RandStream('mt19937ar','Seed',2);
% RandStream.setGlobalStream(seed);


n=200;
p1=30; p2=30; p3=30; 

%% Parameter tensor shape

%Choose the desired signal shape: One brick, Two bricks, 3D Cross, and
%Pyramid (via comment and uncomment the following portions of the code)

% %One brick
% Shape='One brick';
% B=zeros(p1,p2,p3);
% length=5;
% Bcore=ones(length,length,length);
% corner=[15,15,20];
% B(corner(1):corner(1)+length-1,corner(2):corner(2)+length-1,corner(3):corner(3)+length-1)=Bcore;
% r=1; %tensor rank

% %Two bricks
Shape='Two bricks';
B=zeros(p1,p2,p3);
length=5;
Bcore=zeros(length*2,length*2,length*2);
Bcore(1:length,1:length,1:length)=1;
Bcore((length+1):end,(length+1):end,(length+1):end)=1;
corner=[15,15,20];
B(corner(1):corner(1)+2*length-1,corner(2):corner(2)+2*length-1,corner(3):corner(3)+2*length-1)=Bcore;
r=2; %tensor rank
% 
% %3D Cross
% Shape='3D Cross';
% B=zeros(p1,p2,p3);
% block=4;
% signal=1;
% Bcore=zeros(3*block,3*block,3*block);
% Bcore(1:(3*block),(block+1):(2*block),(block+1):(2*block))=signal;
% Bcore((block+1):(2*block),1:(3*block),(block+1):(2*block))=signal;
% Bcore((block+1):(2*block),(block+1):(2*block),1:(3*block))=signal;
% corner=[15,15,15];
% B(corner(1):corner(1)+3*block-1,corner(2):corner(2)+3*block-1,corner(3):corner(3)+3*block-1)=Bcore;
% r=3;
% 
%Pyramid
% Shape='Pyramid';
% B=zeros(p1,p2,p3);
% ht=8; %Change ht for better performance
% if n<=ht*p2
%     r=ht-2;
% else
%     r=ht;
% end
% Bcore=zeros(2*ht-1,2*ht-1,ht);
% for k = 1:r
%     Bcore(k:(ht*2-k),k:(ht*2-k),k)=1;
% end
% corner=[15,15,20];
% B(corner(1):corner(1)+2*ht-2,corner(2):corner(2)+2*ht-2,corner(3):corner(3)+ht-1)=Bcore;

disp('shape B designing complete, loop begins');

%%
% True coefficients for regular (non-array) covariates
p0 = 5;
b0 = ones(p0,1);
sigma = 1;  % noise level

N=200; %N=5 for quick test, N=200 for manuscript
results=zeros(N,4);

for rep=1:N
    fprintf('Replication %d \n',rep);

%Loop starts here
% Simulate covariates
X = randn(n,p0);   % n-by-p0 regular design matrix
Muse = tensor(randn(p1,p2,p3,n));  % p1-by-p2-by-p3-by-n tensor variates
% Simulate responses
mu = X*b0 + double(ttt(tensor(B), Muse, 1:3));
y = mu + sigma*randn(n,1);

%% Voxelwise regression
disp('Voxelwise regression begins');
tic;
Bvox=zeros(p1,p2,p3);
for i1=1:p1
    fprintf('%d ',i1)
    for i2=1:p2
        for i3=1:p3
            Xvox=double(Muse(i1,i2,i3,:));
            if(sum(Xvox)==0)
                Bvox(i1,i2,i3)=0;
            else
                Xtilde=[X,Xvox];
                beta_vox=Xtilde\y;
                Bvox(i1,i2,i3)=beta_vox(p0+1);
            end
        end
    end
end
fprintf('\n VoxelReg finished \n')
toc
results(rep,1)=norm(tensor(B-Bvox));

%% Tensor regression, 3D covariates
% Estimate using Kruskal linear regression - rank = r
disp('Tensor regression begins');
tic;
disp(['TensorReg, rank=', num2str(r)]);
[beta0_rk,beta_rk,glmstat_rk,dev] = kruskal_reg(X,Muse,y,r,'normal');
disp(glmstat_rk{end});
toc
results(rep,2)=norm(tensor(B-double(beta_rk)));


%% Tensor regression with LASSO, 3D covariates
penparam = 1; %no meaning, for Zhou's package use

%% TensorReg with LASSO
disp('Tensor regression with LASSO begins');
LAMBDAL=[1,100,1000];
lassoBIC=inf;
lassoBICpool=zeros(1,size(LAMBDAL,2));
tic;
for s=1:size(LAMBDAL,2)
    lambda_lasso_tmp=LAMBDAL(s); %sqrt(2*log((p1+p2+p3)*r)/n);
    disp(['lambda=', num2str(lambda_lasso_tmp)]);
    [beta0_lasso_tmp,beta_lasso_tmp,~,glmstat_lasso_tmp] = kruskal_sparsereg(X,Muse,y,r,'normal',...
        lambda_lasso_tmp,'lasso',penparam,'B0',beta_rk);
    fprintf('\n');
    lassoBICpool(s)=glmstat_lasso_tmp{end}.BIC;
    if glmstat_lasso_tmp{end}.BIC<lassoBIC
        disp(['lambda=', num2str(lambda_lasso_tmp), ' is selected']);
        beta0_lasso=beta0_lasso_tmp;
        beta_lasso=beta_lasso_tmp;
        glmstat_lasso=glmstat_lasso_tmp;
        lassoBIC=glmstat_lasso_tmp{end}.BIC;
        lambda_lasso=lambda_lasso_tmp;
    end
end
toc
disp(glmstat_lasso{end});
disp(['lambda=', num2str(lambda_lasso), ' is eventually selected']);
results(rep,3)=norm(tensor(B-double(beta_lasso)));

%% SHIV (SHrinkage via Internal Variation)
disp('Shiv begins');
r4=r;
LAMBDA=0.05:0.01:0.09; %2:1:11; %0.5:0.5:5; %or 3:2:23
shivBIC=inf;
shivBICpool=zeros(1,size(LAMBDA,2));
FnormCheat=zeros(1,size(LAMBDA,2));
tic;
for s=1:size(LAMBDA,2)
    lambda_tmp=LAMBDA(s); %sqrt(2*log((p1+p2+p3)*r)/n);
    disp(['lambda=', num2str(lambda_tmp)]);
    [beta0_shiv_tmp,beta_shiv_tmp,~,glmstat_shiv_tmp] = shiv(X,Muse,y,r4,'normal',...
        lambda_tmp,'b1xb2xb3',penparam,'B0',beta_rk);
    fprintf('\n');
    shivBICpool(s)=glmstat_shiv_tmp{end}.BIC;
    FnormCheat(s)=norm(tensor(B-double(beta_shiv_tmp)));
    if glmstat_shiv_tmp{end}.BIC<shivBIC
        disp(['lambda=', num2str(lambda_tmp), ' is selected']);
        beta0_shiv=beta0_shiv_tmp;
        beta_shiv=beta_shiv_tmp;
        glmstat_shiv=glmstat_shiv_tmp;
        shivBIC=glmstat_shiv_tmp{end}.BIC;
        lambda=lambda_tmp;
    end
end
toc
disp(glmstat_shiv{end});
disp(['lambda=', num2str(lambda), ' is eventually selected']);
results(rep,4)=norm(tensor(B-double(beta_shiv)));

end

fprintf('\n')
disp(Shape)
fprintf('Sample size is %d \n',n)
disp(mean(results,1))

save Sim1_2Bricks_200.mat results n p1 p2 p3 Shape r N -v7.3



%Remove results that are 4 standard deviations from the mean
u=mean(results,1);
o=std(results,1);
u_adj=zeros(1,4);
o_adj=zeros(1,4);
for col=1:4
    fprintf('%d data points removed \n',size(results(results(:,col)>=u(col)+4*o(col),col),1));
    u_adj(col)=mean(results(results(:,col)<u(col)+4*o(col),col));
    o_adj(col)=std(results(results(:,col)<u(col)+4*o(col),col));
end
sum(results==0)
disp(num2str(u_adj))
disp(num2str(o_adj))
