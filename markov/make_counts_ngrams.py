import pandas as pd

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

#gives back dictionary for all ngrams of length length_n_grams
#where every ngram will get start_freqs added to amount times seens
def count_ngrams(length_n_grams, start_freqs):    
    
    def make_combinations(l1,l2):
        combinations = []
        for x in l1:
            for y in l2:
                if isinstance(x, (list,)):
                    combinations.append(x + [y])
                else:
                    combinations.append([x,y])
        return combinations
      
        
    all_letters = []
    for x in char_map:
        all_letters.append(x)
    
    #read in excelsheet with ngrams
    data_df = pd.read_excel("ngrams_frequencies_withNames.xlsx")
    data_list = data_df.values.tolist()
    
    #make datalist with right char_map words
    count_sequences = []
    count = [0,0,0,0,0,0,0,0,0,0,0,0]
    for r in data_list:
        new_sequence = []
        letters = r[1].split("_")
        count[len(letters)] += 1
        for l in letters:
            if l == "Tasdi-final":
                l = "Tsadi-final"
            if l == "Tsadi":
                l = "Tsadi-medial"
            new_sequence.append(l)
        new_sequence.reverse()
        count_sequences.append((new_sequence,r[2]))
    
    
    
    #make a list of all possible sequences
    combos = all_letters
    for k in range(length_n_grams - 1):
        combos2 = make_combinations(combos,all_letters)
        combos = combos2
    all_sequences = ["_".join(c) for c in combos]
    
    
    #make dict for recording frequencies
    freqs = dict()
    for c in all_sequences:
        freqs[c] = start_freqs
    
    # For each word in the data
    # for each sequence of n length in word
    # add the amount to sequence of word seen
    for s,c in count_sequences[100:200]:
        if len(s) >= length_n_grams:
            b = 0
            e = length_n_grams
            for p in range(len(s) - length_n_grams + 1):
                current_gram = s[b:e]  
                word_gram = "_".join(current_gram)
                freqs[word_gram] += c
                b += 1
                e += 1
                
    return freqs
        
    
                  
    
    
    
    
    
    
