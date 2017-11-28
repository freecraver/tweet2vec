#!/usr/bin/env python
# encoding: utf-8

import tweet2vec.char
from input.settings_fetch import TRAINING_FILE, VALIDATION_FILE, PATH_MODEL

if __name__ == "__main__":
    # load model-information
    tweet2vec.char.main(TRAINING_FILE, VALIDATION_FILE, PATH_MODEL)