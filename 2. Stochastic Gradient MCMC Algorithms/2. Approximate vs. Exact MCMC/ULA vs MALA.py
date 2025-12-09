import numpy as np
import matplotlib.pyplot as plt

"""
This is an implementation of the example where we want to compare the ULA and MALA schemes for 
a Gaussian posterior distribution with using the Wasserstein-2 distance.
"""

rng = np.random.default_rng(42)

#simulate data according to gaussian distribution
def simulate_data(N_data,var,rng):
    true_theta = rng.normal()
    data = rng.normal(true_theta,np.sqrt(var),size=N_data)
    return true_theta, data

#log of posterior up to normalizing constant ( log p(theta) + log p(y_1:n | theta))
def log_pi(theta,var,data):
     y = np.asarray(data, dtype=float)
     N = y.size
     A = 1 + N/var
     b = y.sum() / var
     return -0.5 * A * theta**2 + b * theta 

#gradient of log posterior
def grad_log_pi(theta,var,data):
    y = np.asarray(data, dtype=float)
    N = y.size
    A = 1 + N/var
    b = y.sum() / var
    return -A * theta + b

#compute closed form parameters of posterior distributions
def true_posterior(data,var):
    y = np.asarray(data, dtype=float)
    N = y.size
    sigma_N = 1/(1 + (N / var))
    mu_N = sigma_N * (y.sum() / var)
    return mu_N, sigma_N

#Unadjusted Langevin Algorithm
def ULA(theta_0,n_iter,step_size,data,var,rng):
    samples = np.empty(n_iter)
    samples[0] = theta_0
    
    for k in range(0,n_iter-1):
        Z = rng.normal()
        samples[k+1] = samples[k] + 0.5*step_size*grad_log_pi(samples[k],var,data) + np.sqrt(step_size)*Z
        
    return samples

#Metropolis Adjusted Langevin Algorithm
def MALA(theta_0,n_iter,step_size,data,var,rng):
    samples = np.empty(n_iter)
    samples[0] = theta_0
    accepts = 0

    # q(theta_new | theta_old) = N(theta_old + delta/2 * grad log pi(theta_old), delta)
    def log_q(theta_new,theta_old):
        mean = theta_old + 0.5 * step_size * grad_log_pi(theta_old, var, data)
        diff = theta_new - mean
        return -0.5 * (diff * diff) / step_size
    
    log_pi_curr = log_pi(samples[0],var,data)
    
    for k in range(0,n_iter-1):
        Z = rng.normal()
        theta = samples[k]
        theta_prop = theta + 0.5*step_size*grad_log_pi(theta,var,data) + np.sqrt(step_size)*Z
        
        log_pi_prop = log_pi(theta_prop,var,data)
        log_alpha = (log_pi_prop + log_q(theta,theta_prop)) - (log_pi_curr + log_q(theta_prop,theta))
        
        if np.log(rng.uniform()) < log_alpha:
            samples[k+1] = theta_prop
            log_pi_curr = log_pi_prop
            accepts += 1
        else: 
            samples[k+1] = theta

    accept_rate = accepts / n_iter    
    return accept_rate, samples

#Wasserstein 2 distance for 1d Gaussians: (mu_a - mu_b)^2 + (std_a - std_b)^2
def w2_gaussian1d(mu_a,sigma_a,mu_b,sigma_b):
    
    return np.sqrt( (mu_a - mu_b)**2 + (np.sqrt(sigma_a) - np.sqrt(sigma_b))**2 )

#running w2 over chain
def running_w2(series, mu_true, var_true, burn=1000, thin=5, every=10):
    means, vars_, its = [], [], []
    buf = []
    for i in range(burn, len(series), thin):
        buf.append(series[i])
        if len(buf) % every == 0:
            m = np.mean(buf)
            v = np.var(buf, ddof=1) if len(buf) > 1 else 0.0
            means.append(m); vars_.append(v); its.append(i)
    w2s = [w2_gaussian1d(m,v, mu_true, var_true) for m,v in zip(means,vars_)]
    return np.array(its), np.array(w2s)


#Simulation

#parameters
var = 4
theta_0 = 0
n_iter = 10000
step_size = 1/1000

#data 
theta_true, data = simulate_data(10000,var,rng)
mu_N , sigma_N = true_posterior(data, var)

#MALA & ULA samples
samples_ULA = ULA(theta_0,n_iter,step_size,data,var,rng)
accept_rate, samples_MALA =  MALA(theta_0,n_iter,step_size,data,var,rng)
print("Acceptance rate for MALA:", accept_rate)

#W2 distances
its_ula, w2s_ula = running_w2(samples_ULA, mu_N, sigma_N)
its_mala, w2s_mala = running_w2(samples_MALA, mu_N, sigma_N)

#plot 
plt.figure(figsize=(6,4))
plt.plot(its_ula, w2s_ula, label="ULA (biased plateau)") 
plt.plot(its_mala, w2s_mala, label="MALA (converges)")
plt.xlabel("iteration") 
plt.ylabel(r"$W_2$ distance to true posterior") 
plt.legend() 
plt.tight_layout()
plt.savefig("ula_vs_mala.png", dpi=300, bbox_inches='tight')
plt.show()




