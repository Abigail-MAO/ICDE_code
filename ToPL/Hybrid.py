import numpy as np
import random


def linear_normalize_p(array,min_val,max_val):
    normalized_array = 2 * (array - min_val) / (max_val - min_val) - 1
    return normalized_array
def linear_denormalize_p(normalized_array,min_val,max_val):
    denormalized_array = 0.5 * (normalized_array + 1) * (max_val - min_val) + min_val
    return denormalized_array

def disturb_data(data,p,q,ee):
    lv=(ee*data-1)/(ee-1)
    rv=(ee*data+1)/(ee-1)
    s=(ee+1)/(ee-1)
    width=rv-lv
    if random.random() < ee/(ee+1):
        return random.uniform(lv, rv)
    else:
        return disturb_O(data,rv,lv,s)



def disturb_O(data,rv,lv,s):
    if random.random() < (data+1)/2:
        data=random.uniform(-s, lv)
    else:
        data=random.uniform(rv, s)
    return data
def pm_data(eps):
    ee = np.exp(eps/2)
    p = (ee /2)*((ee-1)/ (ee + 1))#high probability p
    q = (1 /(2*ee))*((ee-1)/ (ee + 1))#low probability q
    return ee,p,q
def piecewise_h(inputdata,min_val,max_val,epsilon):
    normalized_data=linear_normalize_p(inputdata,min_val,max_val)
    ee,p,q=pm_data(epsilon)
    normalized_data=disturb_data(normalized_data,p,q,ee)
    return np.sum(linear_denormalize_p(normalized_data, min_val, max_val))




def duchi_data(inputdata,epsilon):
    e=np.exp(epsilon)
    p=(e-1)*inputdata/(2*e+2)+0.5
    if random.random()<p:
        return (e+1)/(e-1)
    else:
        return -1*(e+1)/(e-1)

def duchi_h(data, min_val, max_val,epsilon):
    normalized_data = linear_normalize_d(data, min_val, max_val)

    normalized_data = duchi_data(normalized_data,epsilon)
    data_noisy = linear_denormalize_d(normalized_data, min_val, max_val)
    return np.sum(data_noisy)
def linear_normalize_d(array,min_val, max_val):
    normalized_array = 2*(array - min_val) / (max_val - min_val)-1
    return normalized_array
def linear_denormalize_d(normalized_array, min_val, max_val):
    denormalized_array = 0.5*(normalized_array+1) * (max_val - min_val) +min_val
    return denormalized_array




def perturb_select(data,min_val,max_val,epsilon,perturb_p):
    if random.random()<perturb_p:
        return piecewise_h(data,min_val,max_val,epsilon)
    else:
        return duchi_h(data,min_val,max_val,epsilon)
def hybrid(inputdata,min_val,max_val,epsilon):
    inputdata[inputdata > max_val] = max_val
    inputdata[inputdata < min_val] = min_val
    epsilon1=0.61
    outputdata=np.zeros(len(inputdata))
    if epsilon>epsilon1:
        perturb_p=1-np.exp(-epsilon/2)
    else:
        perturb_p=0
    for i in range(len(inputdata)):
        outputdata[i]=perturb_select(inputdata[i],min_val,max_val,epsilon,perturb_p)
    return outputdata


if __name__ == '__main__':
    arr = np.random.rand(10)
    print(arr)
    normalized_arr = 2 * (arr - arr.min()) / (arr.max() - arr.min()) - 1
    print(normalized_arr)


