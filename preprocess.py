"""
Created by John P Cavalieri and Santiago Paredes
"""
from __future__ import print_function

import cPickle as pickle
import collections
import os

import regex
from nltk.probability import FreqDist, ConditionalFreqDist

import modelParameters

TRAINPOS_DIR = './training_data/train/pos'
TRAINNEG_DIR = './training_data/train/neg'
CFDPATH = './model_data/CFD.pickle'


def generate_char_list(string, strip_html=True):
    if strip_html:
        s = strip_html_tags(string.lower())
    else:
        s = string.lower()
    normalized_string = regex.sub(r'\s+', r' ', s)  # change any kind of whitespace to a single space

    list_norm_chars = regex.findall(r"\w|[?!'#@$:\"&*=,]", normalized_string)
    return list_norm_chars


def generate_word_list(string, strip_html=True):
    if strip_html:
        s = strip_html_tags(string.lower())
    else:
        s = string.lower()

    normalized_string = regex.sub(r"\s+", r' ', s)  # change any kind of whitespace to a single space

    # list of words all words seen during training including strings like '!!!' , '??', '....'
    # as these repeated punctuations tend to imply more than the're gramatical meaning
    list_normalized_string = regex.findall(r"\b\w+[']?\w*\b|\!+|\?+|\.{3,}", normalized_string)
    return list_normalized_string


def strip_html_tags(string, verbose=False):
    p = regex.compile(r'<.*?>')
    return p.sub(' ', string)


def sentiment2reviews_map(**kwargs):
    """
    create dictionarys for positive and negative reviews where
    neg_files_map will map a sentiment rating (i.e. 2 stars) to
    a list of reviews with 2 star ratings.  negative review ratings
    range from 1 - 4 and positive review rating 7 - 10.
    e.g. dict['positive'][8][234] will yield the 234'th positive review
    with 8 stars

    Additionally, we also generate (CFD's) conditional frequency distributions
    where CFD['positive']['awesome'] will yield the number of times the word
    'awesome' occurred in positive reviews.These CFDs are just pickled into
    directory model_data for later analysis.


    :return: dict such that dictionary['positive'][8] will be list of
    positive reviews with 8 stars; similarly for negitive reviews
    """

    # reviews are grouped by rating into pos and neg maps as single strings
    pos_files_map = collections.defaultdict(list)  # keys = [7, 8, 9, 10]
    neg_files_map = collections.defaultdict(list)  # keys = [1, 2, 3, 4]

    # note: review_file[-5] gets rating of review from training set
    for review_file in os.listdir(TRAINPOS_DIR):
        with open(os.path.join(TRAINPOS_DIR, review_file), 'r') as review:
            ID_rating = review_file.split('_')
            stars = ID_rating[1].split('.')[0]
            pos_files_map[int(stars)].append((strip_html_tags(review.read())))

    for review_file in os.listdir(TRAINNEG_DIR):
        with open(os.path.join(TRAINNEG_DIR, review_file), 'r') as review:
            ID_rating = review_file.split('_')
            stars = ID_rating[1].split('.')[0]
            neg_files_map[int(stars)].append(strip_html_tags(review.read()))

    sentiment2review_map = {"positive": pos_files_map, "negative": neg_files_map}

    # building cond freq dists

    if not os.path.exists(CFDPATH):
        ratingCFD = ConditionalFreqDist()
        # ratingCFD = collections.defaultdict(Counter)
        for label, file_map in sentiment2review_map.iteritems():
            for stars, reviewList in file_map.iteritems():
                for review in reviewList:
                    ratingCFD[stars].update(FreqDist(generate_word_list(review)))

        with open(CFDPATH, 'wb') as outfile:
            pickle.dump(ratingCFD, outfile)
        # no longer needed once pickled
        del ratingCFD

    return sentiment2review_map


def concat_review_strings(verbose=True):
    """
    concatenates reviews from all reviews, only positive reviews, and
    only negative reviews into large strings used for statistical anlysis


    :return: dict of form {'all':all_reviews, 'positive':pos_reviews, ...}
    but no preprocessing done on review strings except strip html tags.
    """

    if verbose:
        print('concatenting reviews')

    sentimentReview_maps = sentiment2reviews_map()

    # all reviews are concatenated into one single string
    all_reviews = ''
    all_pos_reviews = ''
    all_neg_reviews = ''

    # sentiment2reviews is the dict with keys 'positive' and 'negative'
    for label, file_map in sentimentReview_maps.iteritems():
        for stars, reviewList in file_map.iteritems():
            for review in reviewList:
                all_reviews += review
                if label is 'positive':
                    all_pos_reviews += review
                else:
                    all_neg_reviews += review

    # return dict so that 'all' maps to string of all concatenated reviews ect
    return {'all': all_reviews, 'positive': all_pos_reviews,
            'negative': all_neg_reviews}


def generate_one_hot_maps(vocab_size, skip_top, use_words, DEBUG=None):
    """

    :param vocab_size:
    :param skip_top:
    :param use_words:
    :param verbose:
    :return:
    """

    # assert modelParameters.START_SYMBOL < modelParameters.INDEX_FROM and \
    #     modelParameters.UNK_INDEX < modelParameters.INDEX_FROM, "START_SYMBOL and UNK_INDEX must be < INDEX_FROM"

    # concat_review_strings returns dictionary keyed by 'all', 'positive'
    # and 'negative' so we access 'all' here because we need all_reviews string

    all_reviews = concat_review_strings()['all']

    freq_dist = collections.Counter()

    if use_words:
        freq_dist.update(generate_word_list(all_reviews))

    else:
        freq_dist.update(generate_char_list(all_reviews))
        skip_top = 0

    # list of words starting from most frequent--skip_top removes K most frequent words as stop-word removal technique
    symbols_by_decreasingfreq = freq_dist.most_common()[skip_top:]

    if DEBUG:
        print(symbols_by_decreasingfreq[:70])

    one_hot_maps = dict()
    for idx, (wrd, freq) in enumerate(symbols_by_decreasingfreq[:vocab_size]):
        one_hot_maps[wrd] = idx + modelParameters.INDEX_FROM

    return one_hot_maps


def get_model_data():
    sentrevmap = sentiment2reviews_map()

    allRevDict = concat_review_strings()

    oneHots = generate_one_hot_maps(modelParameters.VocabSize_w, modelParameters.skip_top, True)

    return sentrevmap, allRevDict, oneHots
