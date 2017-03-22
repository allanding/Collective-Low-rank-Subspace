function [ff, ffP]=objHereInner(P,L,Xt,Xs,Zi,Pi,Es,Ei,Qi,Yi,K,d,mu,lambda3)
G0 = 0;
for i = 1:K
    G1 = Pi{i}-P-Es{i}+Yi{i}/mu;
    G2 = Pi{i}'*Xs{i}-P'*Xt*Zi{i}-Ei{i}+Qi{i}/mu;
    %         G0 = G0+mu*sum(sum(G1.^2))+mu*sum(sum(G2.^2));
    G0 = G0+mu*norm(G1,'fro')+mu*norm(G2,'fro');
    
end
G3 = P'*L*P;
ff=G0+lambda3*trace(G3);
M2 = 0;

M1 = mu*eye(d,d)+lambda3*L;
for i = 1:K
    M1 = M1+mu*(K*eye(d)+Xt*Zi{i}*Zi{i}'*Xt');
    M2 = M2-Yi{i}+Xt*Zi{i}*Qi{i}'+mu*(Pi{i}-Es{i}+Xt*Zi{i}*(Xs{i}'*Pi{i}-Ei{i}'));
end
ffP = M1*P-M2;
end