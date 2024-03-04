import math
import numpy as np


# def range(n, alpha,theta, mu,  score_array, epsilon_reminded, epsilon, w):
#     p = 1-math.exp(-1*(alpha/score_array[n]+(1-alpha)*score_array[n]))
#     if epsilon_reminded>epsilon/2:
#         alpha -= 1/(1-alpha)+alpha/((1-alpha)*(1-alpha))
#     elif epsilon/w > epsilon_reminded:
#         alpha += 1/(1-alpha)+alpha/((1-alpha)*(1-alpha))
#     b = math.log(theta/score_array[n]+mu)
#     epsilon_now = p*epsilon_reminded
#     epsilon_reminded = epsilon_reminded- epsilon_now
#     return epsilon_reminded, epsilon_now, b


def perturb(data, n, b, epsilon_now, sample_result):    
    if epsilon_now < 0.000001:
        epsilon_now = 0.000001
    if 1- math.exp(-epsilon_now*b)==0:
        q = 0
    else:
        q = 0.5*epsilon_now/(1-math.exp(-epsilon_now*b))
    # q = 0.5*epsilon_now/(1-math.exp(-epsilon_now*b))
    perturb_value = sample_function(q, b, epsilon_now)
    perturbed_result = data[sample_result[n]]+perturb_value
    # print("value", perturb_value)

    return perturbed_result


def sample_function(a, b, epsilon):
    number = np.random.random_sample()
    if number < 0.5:
        result = math.log(number*a/epsilon+math.exp(-1*epsilon*b))/epsilon
        return result
    else:
        # print(epsilon)
        result = -1*math.log((0.5+a/epsilon-number)*epsilon/a)/epsilon
        return result
