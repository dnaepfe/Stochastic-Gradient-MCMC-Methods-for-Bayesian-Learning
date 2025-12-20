This repository contains the generating code for the Bachelor thesis 'Stochastic Gradient MCMC Methods for Bayesian Learning' by Dominic Näpfer.

Bayesian inference is a popular method for modeling uncertainty in statistical models
and machine learning. Its application, however, is often limited due to the difficulty of
computing posterior distributions. Markov Chain Monte Carlo (MCMC) provides an
approach for sampling from such distributions, but scales poorly to large datasets because of high computational costs. Stochastic Gradient MCMC methods address this
issue by combining stochastic gradient estimates based on minibatches with MCMC
methods, leading to scalable Bayesian inference methods that are well-suited for large
datasets.
This Bachelor thesis presents the mathematical foundations and practical performance
of stochastic gradient MCMC methods for Bayesian learning. In particular, we focus
on stochastic gradient Langevin dynamics (SGLD) and its applications in Bayesian
deep learning. We propose improvements by controlling the stochasticity of the gradient estimator and modifying the randomization strategy used to construct it. For
the empirical results, we implement all methods on a Bayesian neural network and
compare their performance.
