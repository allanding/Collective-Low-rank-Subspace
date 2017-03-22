function P = optimizingP(P,L,Xt,Xs,Zi,Pi,Es,Ei,Qi,Yi,K,d,mu,lambda3,maxIter)
opts.record = 0;
opts.mxitr  = maxIter;
opts.xtol = 1e-3;
opts.gtol = 1e-3;
opts.ftol = 1e-4;
opts.tau = 1e-3;

P=OptStiefelGBB(P, @objHereInner, opts, L,Xt,Xs,Zi,Pi,Es,Ei,Qi,Yi,K,d,mu,lambda3);
end