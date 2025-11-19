import itertools
from collections import Counter

# honestly no clue if this is even remotely close to how this exercises should be done, but i have no other idea
# how to tackle them </3

# todo: theres a logic error here

# EXERCISE 1 (paraphazed): How many colissions will occur where hash function value is 4? Alphabet of 4 chars,
# max length of the word is 3 and modulo is 7

# EXERCISE 4: Determine all collisions when looking for pattern 'ieffd' in text 'gieffdacfhbf', A = { a ... i},
# p = 13

# function that return dictionary of hash function values for 
# each given pattern 
def get_hash_func(albhabet_values: dict, patterns: list, alfabet: list, modulo: int):
    pattern_hash = dict()
    b = len(alfabet)

    for pattern in patterns:
        hash_func = 0
        k = len(pattern)
        for i, char in enumerate(pattern):
            # used a formula i added to notes
            hash_for_char = albhabet_values[char] * pow(b, k-i-1)
            hash_func+=hash_for_char
        hash_func = hash_func % modulo
        pattern_hash.update({pattern : hash_func})

    return pattern_hash


if __name__ == "__main__":

    # EXERCISE 1
    alfabet_values_ex1 = {'a' : 1, 'b' : 2, 'c' : 3, 'd' : 4} # during classes we were indexing from 1 
    alfabet_ex1 = alfabet_values_ex1.keys() # a b c d 

    # find out all possible permutations (with repetive letters) for each word length to get all
    # possible patterns for given alphabet
    perms1_ex1 = [''.join(p) for p in itertools.product(alfabet_ex1, repeat=1)]
    perms2_ex1 = [''.join(p) for p in itertools.product(alfabet_ex1, repeat=2)]
    perms3_ex1 = [''.join(p) for p in itertools.product(alfabet_ex1, repeat=3)] # max word len is 3

    patterns_ex1 = list(itertools.chain(perms1_ex1, perms2_ex1, perms3_ex1))
    modulo_ex1 = 7
    # get hash functions for all these patterns
    pattern_hash_ex1 = get_hash_func(alfabet_values_ex1, patterns_ex1, alfabet_ex1, modulo_ex1)

    # find out which patterns got the same hash function (collisions)
    hash_func_value = 4
    count = 0
    for k, v in pattern_hash_ex1.items():
        if v == hash_func_value:
            print(k)
            count+=1
    print(f'Number of words with the same hashing function ({hash_func_value}) : {count}')

    # EXERCISE 4
    words_to_check = ["gieff", "ieffd", "effda", "ffdac", "fdacf", "dacfh", "acfhb", "cfhbd"] # all possible checks for given text
    alfabet_values_ex4 = {'a' : 1, 'b' : 2, 'c' : 3, 'd' : 4, 'e' : 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9}
    alfabet_ex4 = alfabet_values_ex4.keys()
    modulo_ex4 = 13

    pattern_hash_ex4 = get_hash_func(alfabet_values_ex4, words_to_check, alfabet_ex4, modulo_ex4)
    print(pattern_hash_ex4) # any word with hash func the same as pattern is a collision
