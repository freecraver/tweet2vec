# Define all global params here

# Number of epochs
NUM_EPOCHS = 30
# Batch size
N_BATCH = 64
# Max sequence length
MAX_LENGTH = 145
# Dimensionality of character lookup
CHAR_DIM = 150
# Initialization scale
SCALE = 0.1
# Dimensionality of C2W hidden states
C2W_HDIM = 500
# Dimensionality of word vectors
WDIM = 500
# Number of classes
MAX_CLASSES = 6000
# Learning rate
LEARNING_RATE = 0.01
# Display frequency
DISPF = 5
# Save frequency
SAVEF = 1000
# Regularization
REGULARIZATION = 0.0001
# Reload
RELOAD_MODEL = False
# NAG
MOMENTUM = 0.9
# clipping
GRAD_CLIP = 5.
# use bias
BIAS = False
# use schedule
SCHEDULE = True

# extensions by MF
# validate per user, not per tweet
IS_GROUPED_VALIDATION = True

# add users from validation/test-set iteratively to improve network
USE_ITERATIVE_LEARNING = True

# number of epoch from which on tweets from test-set are added
NUM_EPOCHS_INCLUSION = 6

# percentage of users added to training-set per epoch
PERCENTAGE_ADDED_PER_EPOCH = 0.05

# print additional info
DEBUG_MODE = True