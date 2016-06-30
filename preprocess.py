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

ALL_REVIEWS_PATH = './model_data/all_reviews.pickle'
ALL_POS_REVIEWS_PATH = './model_data/all_pos_reviews.pickle'
ALL_NEG_REVIEWS_PATH = './model_data/all_neg_reviews.pickle'
WORD_ONE_HOT_PATH = './model_data/WORD_one_hot_{}.pickle'
CHAR_ONE_HOT_PATH = './model_data/CHAR_one_hots_{}.pickle'
SENTIMENT_2_REVIEWS_PATH = './model_data/sentmnt2review_map.pickle'

TRAINPOS_DIR = './training_data/train/pos'
TRAINNEG_DIR = './training_data/train/neg'


def generate_char_list(string, strip_html=True):
	if strip_html:
		s = strip_html_tags(string.lower())
	else:
		s = string.lower()
	normalized_string = regex.sub(r'\s+', r' ', s)  # change any kind of whitespace to a single space

	list_norm_chars = regex.findall(r"\w|\s|[?!'#@$:\"&*=,]", normalized_string)
	return list_norm_chars


def generate_word_list(string, strip_html=True):
	if strip_html:
		s = strip_html_tags(string.lower())
	else:
		s = string.lower()

	normalized_string = regex.sub(r'\s+', r' ', s)  # change any kind of whitespace to a single space

	list_normalized_string = regex.findall(r'\b\w+\b|[!?]', normalized_string)  # list of words ('!' and '?' included)
	return list_normalized_string


def strip_html_tags(string, verbose=False):
	p = regex.compile(r'<.*?>')
	return p.sub(' ', string)


def sentiment2reviews_map():
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

	if not os.path.exists(SENTIMENT_2_REVIEWS_PATH):
		# reviews are grouped by rating into pos and neg maps as single strings
		pos_files_map = collections.defaultdict(list)  # keys = [7, 8, 9, 10]
		neg_files_map = collections.defaultdict(list)  # keys = [1, 2, 3, 4]

		# note: review_file[-5] gets rating of review from training set
		for review_file in os.listdir(TRAINPOS_DIR):
			with open(os.path.join(TRAINPOS_DIR, review_file), 'r') as review:
				stars = int(review_file[-5])
				pos_files_map[stars if stars else 10]. \
					append((strip_html_tags(review.read())))

		for review_file in os.listdir(TRAINNEG_DIR):
			with open(os.path.join(TRAINNEG_DIR, review_file), 'r') as review:
				neg_files_map[int(review_file[-5])]. \
					append(strip_html_tags(review.read()))

		sentiment2review_map = {"positive": pos_files_map, "negative": neg_files_map}

		# building cond freq dists
		ratingCFD = ConditionalFreqDist()  # intel python may have nltk bug
		# ratingCFD = collections.defaultdict(Counter)
		for label, file_map in sentiment2review_map.iteritems():
			for stars, reviewList in file_map.iteritems():
				for review in reviewList:
					ratingCFD[stars].update(FreqDist(generate_word_list(review)))

		with open('./model_data/CFD.pickle', 'wb') as outfile:
			pickle.dump(ratingCFD, outfile)
		# no longer needed once pickled
		del ratingCFD

		# the pickled data is now a dictionary keyed by 'positive' and 'negative'
		with open(SENTIMENT_2_REVIEWS_PATH, 'wb') as outfile:
			pickle.dump(sentiment2review_map, outfile)

	else:
		with open(SENTIMENT_2_REVIEWS_PATH, 'rb') as infile:
			sentiment2review_map = pickle.load(infile)

	return sentiment2review_map


def concat_review_strings(verbose=True):
	"""
	concatenates reviews from all reviews, only positive reviews, and
	only negative reviews into large strings used for statistical anlysis


	:return: dict of form {'all':all_reviews, 'positive':pos_reviews, ...}
	but no preprocessing done on review strings except strip html tags.
	"""

	if not os.path.exists(ALL_REVIEWS_PATH):

		if verbose:
			print('concatenting reviews')

		sentiment2review_maps = sentiment2reviews_map()

		# all reviews are concatenated into one single string
		all_reviews = ''
		all_pos_reviews = ''
		all_neg_reviews = ''

		# sentiment2reviews is the dict with keys 'positive' and 'negative'
		for stars in sentiment2review_maps['positive'].keys():
			for reviews in sentiment2review_maps['positive'][stars]:
				for review in reviews:
					all_reviews += review
					all_pos_reviews += review

		for stars in sentiment2review_maps['negative'].keys():
			for reviews in sentiment2review_maps['negative'][stars]:
				for review in reviews:
					all_reviews += review
					all_neg_reviews += review

		# concatenated review strings are pickled
		with open(ALL_REVIEWS_PATH, 'wb') as output_file:
			pickle.dump(all_reviews, output_file)
		with open(ALL_NEG_REVIEWS_PATH, 'wb') as output_file:
			pickle.dump(all_neg_reviews, output_file)
		with open(ALL_POS_REVIEWS_PATH, 'wb') as output_file:
			pickle.dump(all_pos_reviews, output_file)


	else:

		with open(ALL_REVIEWS_PATH, 'rb') as input_file:
			all_reviews = pickle.load(input_file)
		with open(ALL_NEG_REVIEWS_PATH, 'rb') as input_file:
			all_neg_reviews = pickle.load(input_file)
		with open(ALL_POS_REVIEWS_PATH, 'rb') as input_file:
			all_pos_reviews = pickle.load(input_file)

	# return dict so that 'all' maps to string of all concatenated reviews ect
	return {'all': all_reviews, 'positive': all_pos_reviews,
	        'negative': all_neg_reviews}


def generate_one_hot_maps(vocab_size,
                          skip_top,
                          use_words,
                          verbose=False):
	"""
	create mapping of each word or char in Vocab to an integer representing its
	index in a one-hot vector

	:param use_words: if training on characters set False; else defaults to words
	:param verbose:
	:return:
	"""

	if (use_words and not os.path.exists(WORD_ONE_HOT_PATH.format(vocab_size)) or
			    not use_words and not os.path.exists(CHAR_ONE_HOT_PATH.format(vocab_size))):

		if verbose:
			print('building word to 1hot indices mapping')

		# concat_review_strings returns dictionary keyed by 'all', 'positive'
		# and 'negative' so we access 'all' here because we need all_reviews string

		reviewstr_dict = concat_review_strings()
		all_reviews = reviewstr_dict['all']

		if use_words:
			list_normalized_all = generate_word_list(all_reviews)


		else:
			list_normalized_all = generate_char_list(all_reviews)
			# using characters then no need to skip most frequent chars
			skip_top = 0

		_freq_dist = FreqDist(list_normalized_all)

		# list of words starting from most frequent
		# may skip some very freq via skiptop
		# [ skip_top:(vocab_size+skip_top) ]

		words_by_decreasingfreq = (_freq_dist.most_common()[skip_top:(vocab_size + skip_top)])

		if verbose:
			print(words_by_decreasingfreq[20:60])

		one_hot_maps = collections.defaultdict(int)
		for idx, (key, val) in enumerate(words_by_decreasingfreq[:vocab_size]):
			one_hot_maps[key] = idx + modelParameters.UNK_INDEX + 1

		if use_words:
			with open(WORD_ONE_HOT_PATH.format(vocab_size), 'wb') as output_file:
				pickle.dump(one_hot_maps, output_file)
		else:
			with open(CHAR_ONE_HOT_PATH.format(vocab_size), 'wb') as output_file:
				pickle.dump(one_hot_maps, output_file)

	# previously created one_hot_maps data is loaded if it exists
	else:
		if use_words:
			with open(WORD_ONE_HOT_PATH.format(vocab_size), 'rb') as input_file:
				one_hot_maps = pickle.load(input_file)
		else:
			with open(CHAR_ONE_HOT_PATH.format(vocab_size), 'rb') as input_file:
				one_hot_maps = pickle.load(input_file)

	return one_hot_maps


def build_model_data(use_words, vocab_size=None, skip_top=None):
	"""
	need to call this function first before any training or testing
	to specify whether using character based model or word base model and
	then build the data objects required by the model
	:param use_words: True for using words, False for characters
	:param vocab_size: num unique words (or chars)
	:param skip_top: remove the top skip_top most frequent words(chars) as stopwords
	:return:
	"""
	if use_words:
		V = vocab_size if vocab_size is not None else modelParameters.VocabSize_w
	else:
		V = vocab_size if vocab_size is not None else modelParameters.VocabSize_c

	skiptop = skip_top if skip_top is not None else modelParameters.skip_top

	generate_one_hot_maps(V, skiptop, use_words, verbose=True)


if __name__ == '__main__':
	USEWORDS = True
	build_model_data(use_words=USEWORDS)
