"""
Created by John P Cavalieri

"""
trainingCount = 25000
testingCount=11000


#percent of training set to split off into development set
devset_split =14

#size of training and dev combination subsets of pos,neg,mixed reviews for siamese input
trainingComboSize = int((trainingCount * (1 - float(devset_split) / 100)) * float(87) / 43)
devcomboSize = int((trainingCount * float(devset_split) / 100) * float(87) / 43)


#index for words that our out-of-vocab; using 0 may not be the best option
#because many weights in the model will vanish so try small positive ints
UNK_INDEX = 0

# TODO what is sane value for this?
#margin used in contrastive loss function
Margin = 1.25

#assuming full vocab is 85000; much higher in reality
VocabSize_w = 73000  # (73000 +(UNK_INDEX+1)*(UNK_INDEX>0))

#total number unique chars in reviews (no unicode)
VocabSize_c = (58+(UNK_INDEX+1)*(UNK_INDEX>0))

# median review length in words is 2128
MaxLen_w = 540

# 13.5 is median word length of training data
MaxLen_c = (14*MaxLen_w)



#skip the top most frequent words for stop word removal; not valid in character mode
skip_top = 5




