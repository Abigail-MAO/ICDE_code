def pid_control(n, data, time, error_array, sample_result, k_p, k_i, k_d, pi, score_array):
    k = (data[sample_result[n-1]]-data[sample_result[n-2]])/(time[sample_result[n-1]]-time[sample_result[n-2]])
    error_array.append(abs(k*(time[sample_result[n]]-time[sample_result[n-1]])+data[sample_result[n-1]] - data[sample_result[n]]))
    
    # print(n, len(data), sample_result[n-1], sample_result[n])
    score = k_p*error_array[n]
    if n > 0:
        if n-pi-1 >= 0:
            for i in range(n-pi-1, n+1):
                score += k_i/pi*error_array[i]
        else:
            for i in range(0, n + 1):
                score += k_i / pi * error_array[i]
        score += k_d*(error_array[n]-error_array[n-1])/(time[sample_result[n]]-time[sample_result[n-1]])
    score_array.append(max(1.5, score))
    return score_array, error_array
