import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta


"""
We want to model the number of heads of a (possibly) fair coin toss. Here we 
assume a binomial distribution for the likelihood and a beta distribution with
parameters r,s=2 as a prior. Assume we observed x=7 heads out of n=10 trials.
Using the Metropolis-Hasting algorithm, we derive a sample of the posterior 
distribition. 
"""

rng = np.random.default_rng(42)

#prior parameters
r = 2
s = 2

#likelihood parameters (observed data)
n = 10
x = 7

#unnormalized posterior: prior * likelihood
def target_density(theta):
    
    #reject invalid parameters
    if theta <= 0 or theta >= 1:
        return 0 
    
    likelihood = theta**x * (1 - theta)**(n-x)
    prior = theta**(r-1) * (1 - theta)**(s-1)
    
    return likelihood * prior

#Metropolis-Hasting sampler
def metropolis_hastings(num_samples, std):
    
    samples = []
    
    #initial value
    theta = 0.5
    
    for _ in range(num_samples):
        
        # Propose new theta
        theta_proposal = rng.normal(theta, std)
        
        # Compute acceptance probability
        alpha = min(1, target_density(theta_proposal) / target_density(theta))
        
        # Accept or reject
        if rng.uniform() < alpha:
            theta = theta_proposal
        
        samples.append(theta)
    
    return np.array(samples)

# Run the sampler
samples = metropolis_hastings(10000,0.1)

# True posterior
theta_vals = np.linspace(0, 1, 200)
true_pdf = beta.pdf(theta_vals, r + x, s + n - x)

#burn in
burn_in = 1000
post = samples[burn_in:]

#Plots

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

#histogram + true posterior
axes[0].hist(post, bins=40, density=True, alpha=0.5, label='MH samples', color="grey",edgecolor="black", linewidth = 0.5)
axes[0].plot(theta_vals, true_pdf, color="steelblue", lw=2, label='True Beta posterior')
axes[0].set_xlabel(r'$\theta$')
axes[0].set_ylabel('Density')
axes[0].legend()

#trace plot
axes[1].plot(samples, lw=0.8,color="steelblue")
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel(r'$\theta$')

plt.tight_layout()
plt.savefig("mh_example", dpi=300, bbox_inches='tight')
plt.show()

