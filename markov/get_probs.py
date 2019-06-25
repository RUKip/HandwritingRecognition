from make_counts_ngrams import count_ngrams


char_map = {'Alef' : ')', 
            'Ayin' : '(', 
            'Bet' : 'b', 
            'Dalet' : 'd', 
            'Gimel' : 'g', 
            'He' : 'x', 
            'Het' : 'h', 
            'Kaf' : 'k', 
            'Kaf-final' : '\\', 
            'Lamed' : 'l', 
            'Mem' : '{', 
            'Mem-medial' : 'm', 
            'Nun-final' : '}', 
            'Nun-medial' : 'n', 
            'Pe' : 'p', 
            'Pe-final' : 'v', 
            'Qof' : 'q', 
            'Resh' : 'r', 
            'Samekh' : 's', 
            'Shin' : '$', 
            'Taw' : 't', 
            'Tet' : '+', 
            'Tsadi-final' : 'j', 
            'Tsadi-medial' : 'c', 
            'Waw' : 'w', 
            'Yod' : 'y', 
            'Zayin' : 'z'}

def probs(sequence, ngramdic):
    total_hits = 0
    for l in char_map:
        new_sequence = sequence + [l]
        lettered = "_".join(new_sequence)
        total_hits += ngramdic[lettered]
    probs = dict()
    if total_hits > 0:      
        for l in char_map:
            new_sequence = sequence + [l]
            lettered = "_".join(new_sequence)
            probs[l] = ngramdic[lettered] / total_hits
    else:    #if  never seen this sequence before, all letters are even likely
        for l in char_map: 
            new_sequence = sequence + [l]
            lettered = "_".join(new_sequence)
            probs[l] = 1/27
    print("Total times seen this sequence: ", total_hits)
    return probs



counts = count_ngrams(2,0)
print(probs(['Kaf'],counts))

