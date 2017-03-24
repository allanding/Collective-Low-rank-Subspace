function Pt = CLRS(Xs,Ys,Xtt,Ytt,options)
D = [];
Yss = [];
K = options.K;
for i=1:K
    D = [D Xs{1,i}];
    si{i} = size(Xs{1,i},2);
    Yss = [Yss;Ys{i}];
end


%% calculate within-class and between-class terms
count = hist(Yss,unique(Yss));
n = sum(count);
Hw = [];
for i = 1:length(count)
    Hw = blkdiag(Hw,ones(count(i))/count(i));
end
Hb = ones(n)/n-Hw;

[d,t] = size(D);

%% initialize subspace P with random matrix
p = options.ReducedDim;
rand('seed',1)
Pt = rand(d,p);



%% initialize otehr variables
Ji = cell(K,1);
Zi = cell(K,1);
Pi = cell(K,1);
Ei = cell(K,1);
Es = cell(K,1);

Fi = cell(K,1); %% <Fi,Zi-Ji>
Yi = cell(K,1); %% <Yi,Pi-Pt-Es>
Qi = cell(K,1); %% <Qi,Pi'*Xsi - ...>
R = zeros(size(Pt));

%% initialize variables
for i = 1:K
    Ji{i} = rand(t,si{i});
    Zi{i} = rand(t,si{i});
    Fi{i} = rand(t,si{i});
    Pi{i} = rand(size(Pt));
    Yi{i} = rand(size(Pt));
    Ei{i} = rand(p,si{i});
    Es{i} = rand(size(Pt));
    Qi{i} = rand(p,si{i});
end

%% initialize parameters
max_mu = 1e7;
rho = 1.1;
mu = 1e-6;
lambda3 = options.lambda3; %% for supervised term
lambda = 1e-3; %% for |Ei| & |Es|
eta = 1e-3;


%% determine the iterations

optP = options.optP;

if optP ==3
    maxIter = 3;
else
    maxIter = 10;
end

iter = 0;


while iter < maxIter
    iter = iter + 1;
    %% update Pt
    if(iter > 1)
        L = D*Zz*(Hw-Hb+eta*eye(n))*Zz'*D';
        if optP == 1 %% call solution to P without Low-rank constraint
            M2 = 0;
            M1 = mu*eye(d)+lambda3*L;
            for i = 1:K
                M1 = M1+mu*(K*eye(d)+D*Zi{i}*Zi{i}'*D');
                M2 = M2-Yi{i}+D*Zi{i}*Qi{i}'+mu*(Pi{i}-Es{i}+D*Zi{i}*(Xs{i}'*Pi{i}-Ei{i}'));
            end
            Pt = M1\M2;
            Pt = orth(Pt);
        elseif optP == 2 %% call solution to P with Low-rank constraint
            M2 = mu*Qt+R;
            M1 = 2*mu*eye(d)+lambda3*L;
            for i = 1:K
                M1 = M1+mu*(K*eye(d)+D*Zi{i}*Zi{i}'*D');
                M2 = M2-Yi{i}+D*Zi{i}*Qi{i}'+mu*(Pi{i}-Es{i}+D*Zi{i}*(Xs{i}'*Pi{i}-Ei{i}'));
            end
            Pt = M1\M2;
            Pt = orth(Pt);
        elseif optP == 3 %% call solution to P with Gradient Descent Optimization
            addpath('./FOptM')
            Pt = optimizingP(Pt,L,D,Xs,Zi,Pi,Es,Ei,Qi,Yi,K,d,mu,lambda3,90);
        end
    end
    
    
    
    Fz = [];
    for i = 1:K
        %% update Ji{i}
        tmpJ = Zi{i}+Fi{i}/mu;
        [Uj,sigma,Vj] = svd(tmpJ,'econ');
        svp = length(find(sigma>1/mu));
        if svp>=1
            sigma = sigma(1:svp)-1/mu;
        else
            svp = 1;
            sigma = 0;
        end
        Ji{i} = Uj(:,1:svp)*diag(sigma)*Vj(:,1:svp)';
        
        %% update Zi{i}
        temp_Z1 = D'*Pt*Pt'*D+eye(t);
        temp_Z2 = D'*Pt*(Pi{i}'*Xs{i}-Ei{i})+Ji{i}+(-Fi{i}+D'*Pt*Qi{i})/mu;
        Zi{i} = temp_Z1\temp_Z2;
        
        %% update Ei{i}
        temp = Pi{i}'*Xs{i}-Pt'*D*Zi{i}+Qi{i}/mu;
        Ei{i} = solve_l1l2(temp,lambda/mu);
        
        %% update Es{i}
        temp = Pi{i}-Pt+Yi{i}/mu;
        Es{i} = max(0,temp - lambda/mu)+min(0,temp + lambda/mu);
        
        %% update Pi{i}
        temp_Pi1 = Xs{i}*Xs{i}'+eye(size(Xs{i},1));
        temp_Pi2 = Xs{i}*(Zi{i}'*D'*Pt-Ei{i}')+Pt+Es{i}-(Yi{i}+Xs{i}*Qi{i}')/mu;
        Pi{i} = temp_Pi1\temp_Pi2;
        
        
        
        %% update multipliers
        Fi{i} = Fi{i} + mu*(Zi{i}-Ji{i});
        Qi{i} = Qi{i} + mu*(Pi{i}'*Xs{i}-Pt'*D*Zi{i}-Ei{i});
        Yi{i} = Yi{i} + mu*(Pi{i}-Pt-Es{i});
        Gi = Zi{i}-Fi{i}/mu;
        Fz = [Fz Gi];
        
    end
    %% Update Zz
    Bz  = lambda3*D'*Pt*Pt'*D;
    Zz = lyap(inv(Bz),Hw-Hb+eta*eye(n),-inv(Bz)*Fz);
    len = 0;
    for k =1:K
        temp = Zz(:,1+len:si{k}+len);
        Zi{k} = temp;
        len = len + si{k};
    end
    %% call solution to P with Low-rank constraint
    if optP == 2
        %% update Qt
        tmpQ = Pt-R/mu;
        [Uq,sigma,Vq] = svd(tmpQ,'econ');
        svp = length(find(sigma>1/mu));
        if svp>=1
            sigma = sigma(1:svp)-1/mu;
        else
            svp = 1;
            sigma = 0;
        end
        Qt = Uq(:,1:svp)*diag(sigma)*Vq(:,1:svp)';
        
        %% update R
        R = R + mu*(Pt-Qt);
    end
    
    mu = min(max_mu,mu*rho);

    
    
end
