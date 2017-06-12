"""
Created by John P Cavalieri

"""
trainingCount = 25000
testingCount = 11000

WEIGHT_PATH = './model_data/saved_weights'
SPECS_PATH = './model_data/model_specs'

# percent of training set to split off into development set
devset_split = 16

# size of training and dev pairs for siamese model chosen by experimental results
trainingComboSize = int((trainingCount * (1 - devset_split / 100.0)) * 87.0 / 43)
devcomboSize = int((trainingCount * devset_split / 100.0) * 87.0 / 43)

# index that represents start of review
START_SYMBOL = 1

# index for symbols that our out-of-vocab;
UNK_INDEX = 2

# index from which one-hot indices begin
INDEX_FROM = 4

# margin used in contrastive loss_fn function
Margin = 1.70

# vocab size for words
VocabSize_w = 90000

# total number unique chars in reviews (no unicode)
VocabSize_c = 58

# median review length in words is 2128
MaxLen_w = 2120  # 1000

# 13.5 is median word length of training data
MaxLen_c = (14 * MaxLen_w)

# skip the top most frequent words for stop word removal; not valid in character mode
skip_top = 70
