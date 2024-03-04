from tslearn.datasets  import  UCR_UEA_datasets
import numpy as np
import scipy.stats as stats
import sys
import math
import random 
import data_file_generate as dg
import sax_generate as sg
import symbol_distance as sd
from cdifflib import CSequenceMatcher 
from sklearn.metrics.cluster import adjusted_rand_score as ar
from sklearn.metrics.cluster import rand_score as rs
from sklearn.metrics import normalized_mutual_info_score as nmi

def match_clustering(sequences, data, label, symbol_size):
    count = 0
    symbol_dist = sd.get_distance(symbol_size)
    pred_label = []
    for index in range(len(data)):
        # if label[index] == 1:
        #     continue
        dist = [sd.euclidean_distance(symbol_dist, data[index], seq) for seq in sequences]
        if len([lambda x: x==min(dist), dist]) > 1:
            min_dist = min(dist)
            max_s_value = 0
            can_index = 0
            for i in range(len(dist)):
                # if min_dist == dist[i] and data[index][0] == sequences[i][0]:
                if min_dist == dist[i]:
                    s_value = CSequenceMatcher(None, data[index], sequences[i]).ratio()
                    if max_s_value<s_value:
                        max_s_value = s_value
                        can_index = i
            result = can_index+1
        else:
            # print(data[index], label[index], dist)
            result = dist.index(min(dist))+1
        pred_label.append(result)
        if result == label[index]:
            count += 1
        # else:
        #     print(data[index], label[index], result, dist)
    # print(count, len(data), len(label), count/len(data))
    return count/len(data), ar(label, pred_label), rs(label, pred_label)
    # print(sequences)
    # print()


if __name__ == '__main__':
    # sequences = [ 'cdabc', 'dabcd',  'abcdc']
    # data, label = data_generate(4, 10)
    # match_clustering(sequences, data, label)
    # s1 = CSequenceMatcher(None, ' abcd', 'bcde')
    # s2 = CSequenceMatcher(None, ' abcd', 'abc')
    # print(s1.ratio(), s2.ratio())
    data = [[-0.018914469501158368,-0.026148671378902383,-0.03254566643271929,-0.04203920994362306,-0.0486907285614113,-0.050534972642137785,-0.056846005015248,-0.059099709219002754,-0.06110552029867699,-0.06744992719578019,-0.07002263091853528,-0.07256996652561234,-0.07382168373647612,-0.07643120791902086,-0.08194545876439134,-0.08400483746852654,-0.085301358806898,-0.09007890113251452,-0.09349873588467544,-0.09312725152628917,-0.0991323199824177,-0.10045651454438764,-0.10335015162254167,-0.10536614206032185,-0.10814587919172598,-0.1104595860816816,-0.07861712807442145,-0.013892536653380517,0.054006370642961786,0.10552699250050043,0.12957417910417332,0.059982170638582515,-0.11984824341267786,-0.23492463798229737,-0.2521058538468447,-0.25126672423197466,-0.23881059516172906,-0.2375036745534833,-0.2475664073328276,-0.24764756814617694,-0.23744178590551737,-0.22534813065491544,-0.2116544003660341,-0.19919682532913177,-0.19316587961792409,-0.15410005643708577,0.29468185485770154,1.5435322424759683,0.8527765541535418,-2.365445660195318,-5.255331366519016,-5.298602675669611,-3.5038574244436917,-1.8822090854022866,-0.8790927821116895,-0.40082059489191385,-0.1769364815553154,-0.07035851357898411,-0.006746091203820866,0.034872522532161526,0.06571324748826976,0.09355296175306614,0.1202250121315666,0.15233347024527769,0.1938144691644395,0.23328206545445532,0.2867287828591777,0.34583193369830006,0.42408114810674763,0.5195356596177463,0.6401447249088051,0.795998371732295,0.9888169740869667,1.2311772953553053,1.5134759004406835,1.8210642415434435,2.128755201696212,2.4191075738256806,2.650342260736244,2.753495948379836,2.647396258113402,2.2986106248731555,1.7694259921524063,1.1955591700140331,0.7074705820550163,0.3578999892878551,0.14535999636991964,0.031028418686572364,-0.03384344265789881,-0.05890438215900545,-0.06504145209299615,-0.06388150187024469,-0.05449643324998511,-0.04251805845047131,-0.028503689451430055,-0.012943390082137423,-0.005175083315245946,-0.000979258131709112,-0.005072375355932821,-0.008607292576376829,-0.021161716611321366,-0.039125156739770475,-0.05689218697414249,-0.0747631224120984,-0.09511249959419231,-0.11333879823696168,-0.13215041624338897,-0.15242382571406585,-0.16747219549293677,-0.1798078780800849,-0.18964041376932753,-0.1936829557261766,-0.20685841410025815,-0.21708686407183256,-0.2250753844697098,-0.23332472927340608,-0.2408334647914343,-0.24544066996002425,-0.25016240495722747,-0.25531839187085653,-0.2591971211690215,-0.26333102739746533,-0.2647565252913421,-0.26829847030166465,-0.2683008241470119,-0.2731690936276892,-0.2799461516755911,-0.2773080228253271,-0.24160472852981435,-0.2140019518574566,-0.17418367499478873,-0.13635059302217056,-0.08601071678626707,-0.0356754687838525,-0.0009222193026013488,0.00109191],
[0.01806283029333852,0.027018379538381713,0.0488630585996334,0.05740120084338092,0.06298020483087675,0.06383208901579208,0.061406603432245625,0.056899979757912146,0.04984573412959375,0.04173001651107738,0.030334338179383823,0.0203153860473244,0.0104121119746956,-0.00231903039865361,-0.009539462376623419,-0.01619093356041294,-0.020478271345423585,-0.027145824430499326,-0.0348981605076575,-0.0358099998655935,-0.04077310539549941,-0.043092114504663125,-0.04496761734166923,-0.04644296867740554,-0.04911462470439697,-0.04913538599016948,-0.053058925062983736,-0.05350489933472249,-0.05541928828670657,-0.053958173614702426,-0.03873543033299283,0.019346209022696393,0.09718679523077896,0.16099263262600028,0.2089728871771259,0.2211210699907256,0.09089490883149337,-0.055468820986332094,-0.15540092217051032,-0.20707590648949167,-0.2197067206537297,-0.2076166083480962,-0.19256561067203426,-0.19360647892095256,-0.188740867553217,-0.17942953222432695,-0.16610281255040735,-0.1492488507228413,-0.13643679084324095,-0.11743853429564843,0.07354604828798937,1.2084931082039339,2.3653394382442134,0.09446864307409004,-3.918340683800915,-6.275626351717405,-5.354819926195672,-3.442697676524674,-1.8655886542473683,-0.8816051968190979,-0.3970135136164563,-0.17405264269837006,-0.073525868097707,-0.01349242577986911,0.025660116815258587,0.05865949143984369,0.09169040352278943,0.12874748259327873,0.1728085216206946,0.2296579336448201,0.29846879005933524,0.38806443087848047,0.5030095187887327,0.6523703614050752,0.8418578353429207,1.0674211706368222,1.31610259189809,1.5632876424746613,1.7790266680730582,1.9450841428264962,2.0235999379036094,1.9701577765922063,1.7451780618200994,1.3857458291932983,0.9805242259750998,0.6242844145103419,0.36688404165663774,0.19818379165236627,0.0987117970586132,0.03708731374619247,0.006262886764316656,-0.009633705990028149,-0.01329355563266663,-0.007427870112221252,-0.0010459921224391696,0.01327895643253445,0.03137631402198848,0.04833192398525038,0.058758524731276185,0.06400277199318524,0.06354147887222565,0.06496360302875831,0.058735381666049594,0.05006918317118888,0.0364558135797862,0.022584052366494457,0.011647568723202713,-0.0008893065128353486,-0.011813715680958181,-0.019945946646710566,-0.028494896368151636,-0.03443175554975417,-0.04271901325165727,-0.047241998224673336,-0.0498625635625461,-0.053593072815793515,-0.056096862018338835,-0.057601744291868956,-0.05936615022614204,-0.062301063956294665,-0.06444985191133186,-0.06470581925643554,-0.06473110176160869,-0.06063855581262921,-0.04936479211120843,-0.024354278620686832,0.021963170270704713,0.07373968097426248,0.11323401715731644,0.12012439376944442,0.08369590947618435,0.004740905758114373,-0.07123828901221006,-0.11846382882216062,-0.11574508000375586,-0.09960808]]
    # print(sg.sax_delete_repeat(sg.sax(data[0], 10, 10)))
    # print(sg.sax_delete_repeat(sg.sax(data[1], 10, 10)))
    # sys.exit()
    window_length = 10
    symbol_size = 10
    import process as eg
    path = '../data/'
    test_data = eg.read_data(path+'/test_data.txt')
    test_label = eg.read_data(path+'/test_label.txt', label=True)
    
    result = match_clustering(['aacdddccbbaaa', 'aabdddcccbbaa'], test_data, test_label, symbol_size)
    print(result)
