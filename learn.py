#!/usr/bin/env python
# encoding: utf-8

import tweet2vec.char
import tweet2vec.encode_char
import tweet2vec.test_char
import tweet2vec.evaluate
from input.settings_fetch import TRAINING_FILE, VALIDATION_FILE, PATH_MODEL, PATH_RESULT

if __name__ == "__main__":
    # load model-information
    #tweet2vec.char.main(TRAINING_FILE, VALIDATION_FILE, PATH_MODEL)
    # predict validation data
    tweet2vec.test_char.main([VALIDATION_FILE, PATH_MODEL, PATH_RESULT])
    #for i in range(1,11):
    #    tweet2vec.evaluate.main(PATH_RESULT + "\\" + str(i) + ". Iteration", PATH_MODEL)
    #tweet2vec.evaluate.main(PATH_RESULT,PATH_MODEL)
    # encode tweets
    #tweet2vec.encode_char.main([VALIDATION_FILE, PATH_MODEL, PATH_RESULT])