#!/usr/bin/env python
# coding: utf-8

"""
Spelling related functions
"""

from nltk.corpus import words as nltk_words
word_dict = {w:1 for w in nltk_words.words()}

def is_typo(word):
    if 'param_' in word \
    or word.lower() in word_dict \
    or word.capitalize() in word_dict:
        return False
    else:
        return True


def my_spell(word):
    """
    if word is a typo, this function attemps to correct it,
    if that fails word is repleced by an empty string
    """
    from autocorrect import spell
    
    corrected_word = ''
    rescued_typo = 0
    
    if len(word)>1: # one letter word are not considered 
        
        # try to correct typo
        if is_typo(word): 
            print('typo: ' + word)
            word = spell(word)
            print('autocorrected typo: ' + word)

            if not is_typo(word): 
                rescued_typo = 1
                corrected_word = word
        else:
            corrected_word = word

    return corrected_word, rescued_typo
