import math
import numpy as np
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

""" 
This is an extended implementation of the Bayesian neural network model for 
classification of the MNIST handwritten digits dataset we used before. Here we 
use different randomization strategies (RR and RM) for generating the posterior
sample with SGLD and compare them on different metrics. 
"""

torch.manual_seed(42)
random.seed(42)

#Two layer neural network (A=10x100,a=1x10,B=784x100,b=1x100)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 100, bias=True) #(B,b)
        self.fc2 = nn.Linear(100, 10,  bias=True) #(A,a)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)       
        return x
    
#-----------------Optimizer-----------------------------------------------
#SGLD Optimizer
class SGLD(optim.Optimizer):
    def __init__(self, params, lr=None):
        super().__init__(params, dict(lr=lr))

    @torch.no_grad()
    def step(self):
        
        for group in self.param_groups:
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Langevin noise ~ N(0, 2*lr*I)
                noise = torch.randn_like(p) * math.sqrt(2.0 * lr)

                # Parameter update 
                p.add_(-lr * p.grad + noise)
                        
#--------------------------- Train ------------------------------------------
#Train using RR or RM
def train(model, train_loader, num_it, lr, burn_in, sample_every, rand_mode):
    
    optimizer = SGLD(model.parameters(), lr=lr)     
    data_size = len(train_loader.dataset)  # N
    batch_size = train_loader.batch_size   # n
    scale = data_size / batch_size         # R

    samples = []
    it = 0
    epoch = 0
    print(f"Starting SGLD ({rand_mode}) | target updates: {num_it} | batches/epoch: {len(train_loader)}")

    # Random Reshuffling m = data_size / batch_size
    if rand_mode == "RR":

       while it < num_it:
           model.train()
           epoch += 1
           # iterates through all batches and reshuffles after every epoch
           for x, y in train_loader:
               if it >= num_it:
                   break

               optimizer.zero_grad(set_to_none=True)
               output = model(x)
            
               #likelihood loss
               data_loss = F.cross_entropy(output, y)
               #prior loss θ ~ N(0, I)
               prior_loss = 0.5 * sum(p.pow(2).sum() for p in model.parameters())
               #full loss
               total_loss = scale * data_loss + prior_loss
            
               total_loss.backward()
               optimizer.step()

               it += 1

               # collect samples
               if it > burn_in and ((it - burn_in) % sample_every == 0):
                   samples.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

           print(f"Epoch {epoch} | updates {it}/{num_it} | samples {len(samples)}")
           
    # Robbins-Monro m = 1
    if rand_mode == "RM":
         
        batches = list(train_loader)  
        
        while it < num_it:
            model.train()
            epoch = it // scale
            
            #pick one batch
            x,y = random.choice(batches)
            
            optimizer.zero_grad(set_to_none=True)
            output = model(x)
         
            #likelihood loss
            data_loss = F.cross_entropy(output, y)
            #prior loss θ ~ N(0, I)
            prior_loss = 0.5 * sum(p.pow(2).sum() for p in model.parameters())
            #full loss
            total_loss = scale * data_loss + prior_loss
         
            total_loss.backward()
            optimizer.step()
            
            it += 1
            
            # collect samples
            if it > burn_in and ((it - burn_in) % sample_every == 0):
                samples.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
                
            if it % 500 == 0:     
                print(f"Epoch {epoch} | updates {it}/{num_it} | samples {len(samples)}")

    print(f"Done. Collected {len(samples)} samples.")
    return samples

#--------------------------- Evaluate -----------------------------------------
def adaptive_calibration_error(mean_probs, targets):
    
    num_bins = 15
    
    with torch.no_grad():
        confidences, predictions = mean_probs.max(dim=1)
        correct = (predictions == targets).float()
        
        sorted_conf, indices = torch.sort(confidences)
        sorted_correct = correct[indices]

        N = mean_probs.size(0)
        bin_size = max(1, N // num_bins)

        bin_confs = []
        bin_accs  = []

        ace = 0.0
        
        for b in range(num_bins):
            start = b * bin_size
            end   = (b + 1) * bin_size if b < num_bins - 1 else N
            if start >= end:
                continue

            bin_conf = sorted_conf[start:end]
            bin_corr = sorted_correct[start:end]

            avg_conf = bin_conf.mean().item()
            avg_acc  = bin_corr.mean().item()

            bin_confs.append(avg_conf)
            bin_accs.append(avg_acc)

            # ACE contribution
            weight = (end - start) / N
            ace += weight * abs(avg_acc - avg_conf)

        return (
            np.array(bin_confs),
            np.array(bin_accs),
            ace,
        )
        
#Evulation using Bayesian predictive mean (E[f(x_test,θ)] = Ave (f(x_test,θ_samples)) with using softmax
# returns accuracy, negative log-likelihood, adaptive calibration error, probabilities and targets
def evaluate(model,samples,data_loader):
    model.eval()
    
    full_mean_probs = []
    full_labels = []
    
    with torch.no_grad():
    
       for x, y in data_loader:
        
           probs_sum = None
        
           for sample in samples:
               model.load_state_dict(sample, strict=True)
               output = model(x)
               probs = torch.softmax(output,dim=1)
               probs_sum = probs if probs_sum is None else (probs_sum + probs)
        
           #compute average 
           probs_mean = probs_sum / len(samples)
           full_mean_probs.append(probs_mean.cpu())
           full_labels.append(y.cpu())
        
    
    mean_probs = torch.cat(full_mean_probs, dim=0)
    targets = torch.cat(full_labels, dim=0)         
    
    #accuracy
    preds = mean_probs.argmax(dim=1)
    acc = (preds == targets).float().mean().item()
    
    #negative log-likelihood
    true_class_probs = mean_probs[torch.arange(mean_probs.size(0)), targets]
    nll = -true_class_probs.log().mean().item()
    
    #adaptive calibration error
    bin_confs, bin_accs, ace = adaptive_calibration_error(mean_probs, targets)
    
    results = {"acc": acc, 
               "nll": nll,
               "ace": ace, 
               "bin_confs": bin_confs,
               "bin_accs": bin_accs,
               "mean_probs": mean_probs, 
               "targets": targets}
    
    return results
    
#------------------------------- Plots ------------------------------------
def metric_over_samples(data_loader,samples_RR,samples_RM,metric):
    
    model = Net()
    
    num_samples = min(len(samples_RR), len(samples_RM))
    sample_sizes = list(range(10, num_samples + 1, 10))

    RR_vals = []
    RM_vals = []

    for sub_samples in sample_sizes:
        subset_RR = samples_RR[:sub_samples]
        subset_RM = samples_RM[:sub_samples]

        res_RR = evaluate(model, subset_RR, data_loader)
        res_RM = evaluate(model, subset_RM, data_loader)
                          
        RR_vals.append(res_RR[metric])
        RM_vals.append(res_RM[metric])

        print(
            f"Samples: {sub_samples:3d} | "
            f"RR {metric}: {res_RR[metric]:.4f} | "
            f"RM {metric}: {res_RM[metric]:.4f}"
        )

    return sample_sizes, RR_vals, RM_vals

def plot_metric_vs_samples(sample_sizes, RR_vals, RM_vals, ylabel, filename):
    plt.figure(figsize=(6, 6))
    plt.plot(sample_sizes, RR_vals, color='darkblue', marker='+', label="RR", linewidth=1)
    plt.plot(sample_sizes, RM_vals, color='darkgreen', marker='+', label="RM", linewidth=1)
    plt.xlabel("Number of samples")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout() 
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    
def accuracy_plot(samples_RR, samples_RM, test_loader):
    sample_sizes, RR_accs, RM_accs = metric_over_samples(test_loader, samples_RR, samples_RM, metric="acc")
    plot_metric_vs_samples(sample_sizes, RR_accs, RM_accs, ylabel="Accuracy", filename="BNN_RR_RM_acc.png")

def nll_plot(samples_RR, samples_RM, test_loader):
    sample_sizes, RR_nlls, RM_nlls = metric_over_samples(test_loader, samples_RR, samples_RM, metric="nll")
    plot_metric_vs_samples(sample_sizes, RR_nlls, RM_nlls, ylabel="Negative log-likelihood", filename="BNN_RR_RM_nll.png")

def ace_plot(samples_RR, samples_RM, test_loader):
    sample_sizes, RR_aces, RM_aces = metric_over_samples(test_loader, samples_RR, samples_RM,metric="ace")
    plot_metric_vs_samples(sample_sizes, RR_aces, RM_aces,ylabel="Adaptive Calibration Error",filename="BNN_RR_RM_ace.png")
    
def plot_reliability_diagram(results):
    plt.figure(figsize=(6, 6))
    colors = ['darkblue','darkgreen']

    for i, (name, res) in enumerate(results.items()):
        mean_probs = res["mean_probs"]
        targets = res["targets"]
        bin_confs, bin_accs, _ = adaptive_calibration_error(mean_probs, targets)
        plt.plot(bin_confs, bin_accs, color = colors[i], marker="+", label=name)

    # perfect calibration line
    plt.plot([0, 1], [0, 1], "--")

    plt.xlabel("Mean predicted confidence")
    plt.ylabel("Empirical accuracy")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig("BNN_RR_RM_rd", dpi=300, bbox_inches="tight")
    plt.show()


#------------------------Simulation---------------------------------------------
#Load data 
train_data = datasets.MNIST(root='data',train=True, download=True, transform=ToTensor()) 
test_data = datasets.MNIST(root='data',train=False, download=True, transform=ToTensor()) 

batch_size = 120
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader (test_data, batch_size=batch_size, shuffle=True)

model = Net()

#generate samples
samples_RR = train(model, train_loader, num_it=20000, lr=1e-4, burn_in=5000, sample_every=100, rand_mode="RR")
samples_RM = train(model, train_loader, num_it=20000, lr=1e-4, burn_in=5000, sample_every=100, rand_mode="RM")

#evaluate
results_RR = evaluate(model, samples_RR, test_loader)
results_RM = evaluate(model, samples_RM, test_loader)

results = {"RR": results_RR, "RM": results_RM}

#plots
plot_reliability_diagram(results)
accuracy_plot(samples_RR, samples_RM, test_loader)
nll_plot(samples_RR, samples_RM, test_loader)
ace_plot(samples_RR, samples_RM, test_loader)

