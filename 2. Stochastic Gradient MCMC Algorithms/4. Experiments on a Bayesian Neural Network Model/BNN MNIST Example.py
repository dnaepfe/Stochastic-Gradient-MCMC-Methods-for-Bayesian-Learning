import math
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

""" 
This is an implementation of a Bayesian neural network model for 
classification of the MNIST handwritten digits dataset and using 
SGLD to compare the predictive accuracy and quantify uncertainty. 
"""

torch.manual_seed(42)

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
    
#-----------------Optimizer-------------------------------------------
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
                        
#------------------------Train & Evaluate----------------------------------------
#Train & aprroximate sample from posterior
def train(model, train_loader, num_it, lr, burn_in, sample_every):
    
    optimizer = SGLD(model.parameters(), lr=lr)     
    data_size = len(train_loader.dataset)
    batch_size = train_loader.batch_size
    scale = data_size / batch_size

    samples = []
    it = 0
    epoch = 0
    print(f"Starting SGLD | target updates: {num_it} | batches/epoch: {len(train_loader)}")

    while it < num_it:
        model.train()
        epoch += 1
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

    print(f"Done. Collected {len(samples)} samples.")
    return samples

#Evulation using Bayesian predictive mean (E[f(x_test,θ)] = Ave (f(x_test,θ_samples)) with using softmax
def evaluate(model,samples,data_loader):
    model.eval()
    
    full_mean_probs = []
    full_labels = []
    
    for x, y in data_loader:
        
        probs_sum = 0 
        
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
    preds = mean_probs.argmax(dim=1)
    acc = (preds == targets).float().mean().item()

    return mean_probs, acc

#---------------------Plots-----------------------------------------------------
#plot test examples with predictive uncertanty
def uncertanty_plot(model, samples, test_loader, num_examples):
    
    torch.manual_seed(100)
    model.eval()
    
    #test examples
    test_iter = iter(test_loader)
    x_batch, y_batch = next(test_iter)
    
    fig, axes = plt.subplots(2, num_examples, figsize=(15, 6))
    
    for i in range(num_examples):
        x = x_batch[i:i+1]
        true_label = y_batch[i].item()
        
        # Collect predictions from all samples
        sample_predictions = []
        with torch.no_grad():
            for sample in samples:
                model.load_state_dict(sample)
                output = model(x)
                probs = F.softmax(output, dim=1)
                sample_predictions.append(probs.cpu().numpy()[0])
        
        sample_predictions = np.array(sample_predictions)
        
        # Top: image
        axes[0, i].imshow(x_batch[i].squeeze(), cmap='gray')
        axes[0, i].set_title(f'True: {true_label}')
        axes[0, i].axis('off')
        
        # Bottom: predictive distribution
        mean_probs = np.mean(sample_predictions, axis=0)
        std_probs = np.std(sample_predictions, axis=0)
        
        axes[1, i].bar(range(10), mean_probs, yerr=std_probs, 
                      alpha=0.7, capsize=3, color='steelblue', 
                      error_kw=dict(elinewidth=1, ecolor='red'))
        axes[1, i].set_xticks(range(10))
        axes[1, i].set_ylim(0, 1)
        axes[1, i].set_title(f'Pred: {np.argmax(mean_probs)}')
        axes[1, i].set_xlabel('Digit')
        if i == 0:
            axes[1, i].set_ylabel('Probability')
    
    plt.tight_layout()
    plt.savefig("BNN_uncertanty_plot", dpi=300, bbox_inches='tight')
    plt.show()
    
#plot predictive accuracy
def accuracy_plot(samples, train_loader, test_loader):
    
    torch.manual_seed(100)
    model = Net()
    num_samples = len(samples)
    sample_sizes = list(range(10, num_samples + 1, 10))
    train_accs = []
    test_accs = []
    
    for sub_samples in sample_sizes:
        if sub_samples > num_samples:
            continue
        
        #compute accuracy for sub samples
        subset = samples[:sub_samples]
        train_outputs, train_acc = evaluate(model, subset, train_loader)
        test_outputs, test_acc = evaluate(model, subset, test_loader)
        
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        print(f"Samples: {sub_samples:2d} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes[:len(train_accs)], train_accs, color='steelblue', label='Training Accuracy', linewidth=2)
    plt.plot(sample_sizes[:len(test_accs)], test_accs, color='darkorange', label='Test Accuracy', linewidth=2)
    plt.xlabel('Number of Samples')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig("BNN_accuracy_plot", dpi=300, bbox_inches='tight')
    plt.show()
    
#------------------------Simulation---------------------------------------------
#Load data 
train_data = datasets.MNIST(root='data',train=True, download=True, transform=ToTensor()) 
test_data = datasets.MNIST(root='data',train=False, download=True, transform=ToTensor()) 

batch_size = 128
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader (test_data, batch_size=batch_size, shuffle=True)

model = Net()

#generate samples
samples = train(model, train_loader, num_it=20000, lr=1e-4, burn_in=5000, sample_every=100)

#plot
uncertanty_plot(model, samples, test_loader, num_examples=4)
accuracy_plot(samples, train_loader, test_loader)



    

