import numpy as np 
import sax_generate as sg
import symbol_distance as sd

ground_truth_center = np.load('../ground_truth/ground_truth_center.npy')
patternLDP_center = np.load('../patternLDP/patternLDP_center.npy')

symbol_size = 6
paa_length = 25

ground_truth_sax = [sg.sax(elem, symbol_size, paa_length) for elem in ground_truth_center]
ground_truth_sax = [sg.sax_delete_repeat(sax_sequence) for sax_sequence in ground_truth_sax]

patternLDP_sax = [sg.sax(elem, symbol_size, paa_length) for elem in patternLDP_center]
patternLDP_sax = [sg.sax_delete_repeat(sax_sequence) for sax_sequence in patternLDP_sax]

symbol_dist = sd.get_distance(symbol_size)
dist_dtw = sum([sd.similar_match_dtw(symbol_dist, ground_truth_sax[i], patternLDP_sax[i]) for i in range(len(ground_truth_sax))])
dist_sed = sum([sd.similar_match(symbol_dist, ground_truth_sax[i], patternLDP_sax[i]) for i in range(len(ground_truth_sax))])
dist_euc = sum([sd.euclidean_distance(symbol_dist, ground_truth_sax[i], patternLDP_sax[i]) for i in range(len(ground_truth_sax))])

print(dist_dtw, dist_sed, dist_euc)
# 2023 epsilon 4
# 38.97 10.11 46.3
        
        
        
        
        