import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def reweighting_stats(penalties,beta):
    """
    post re-weighting stats
    penalties: arbitrage penalties #days x #samples
    """
    data_m = penalties
    transform = np.exp(- beta * penalties)
    for i in range(transform.shape[0]):
        transform[i,:] = transform[i,:] / np.sum(transform[i,:])
    mean_before = np.mean(data_m,axis = 1)
    median_before = np.median(data_m, axis = 1)
    mean_after = np.zeros(transform.shape[0])
    median_after = np.zeros(transform.shape[0])
    for i in tqdm(range(transform.shape[0])):
        mean_after[i] = np.sum(transform[i,:] * data_m[i,:])
        args = np.argsort(data_m[i])
        data_m[i,:] = data_m[i,args]
        transform[i,:] = transform[i,args]
        med = data_m[i,0]
        sm = 0
        j = 0
        while sm < 0.5:
            sm = sm + transform[i,j]
            med = data_m[i,j]
            j = j +1
        median_after[i] = med
    print("Mean mean before ", np.mean(mean_before)," and after reweighting ",np.mean(mean_after))
    print("std of means before ", np.std(mean_before)," and after reweighting across time ", np.std(mean_after))
    print("Mean median before ", np.median(mean_before)," and median of means", np.median(mean_after))
 
    plt.hist(data_m[1,:],bins=50,density=True,weights=transform[1,:],color='blue',label='Reweighted data')
    plt.hist(data_m[1,:],bins=50,density=True,color='red',label='Original data')
    plt.legend(loc='upper center')
    plt.title("Arbitrage violations of generated data on a sample day")
    plt.show()
    plt.hist(mean_after,bins=50,density=True,color='blue',label='Reweighted data')
    plt.hist(mean_before,bins=50,density=True,color='red',label='Original data')
    plt.legend(loc='upper right')
    plt.title("Mean mean arbitrage violations of generated data")
    plt.show()
    plt.hist(median_after,bins=50,density=True,color='blue',label='Reweighted data')
    plt.hist(median_before,bins=50,density=True,color='red',label='Original data')
    plt.legend(loc='upper right')
    plt.title("Mean medians of arbitrage violations of generated data")
    plt.show()
    return data_m, transform

def reweighting_stats_mean(penalties,beta):
    """
    post re-weighting stats (mean)
    penalties: arbitrage penalties #days x #samples
    """
    data_m = penalties
    transform = np.exp(- beta * data_m)
    for i in range(transform.shape[0]):
        transform[i,:] = transform[i,:] / np.sum(transform[i,:])

    mean_after = np.zeros(transform.shape[0])
    for i in tqdm(range(transform.shape[0])):
        mean_after[i] = np.sum(transform[i,:] * data_m[i,:])
    median_after = np.zeros(transform.shape[0])
    for i in tqdm(range(transform.shape[0])):
        mean_after[i] = np.sum(transform[i,:] * data_m[i,:])
        args = np.argsort(data_m[i])
        data_m[i,:] = data_m[i,args]
        transform[i,:] = transform[i,args]
        med = data_m[i,0]
        sm = 0
        j = 0
        while sm < 0.5:
            sm = sm + transform[i,j]
            med = data_m[i,j]
            j = j +1
        median_after[i] = med
    
    return mean_after, median_after