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
# validate per tweet; per average classification of user; per nr. of max classifications of user
GV_NONE, GV_AVERAGE, GV_ARGMAX = range(0,3)
GROUP_VALIDATION_MODE = GV_ARGMAX
IS_GROUPED_VALIDATION = (GROUP_VALIDATION_MODE != GV_NONE)

# print additional info
DEBUG_MODE = True

# adaptive learning parameters
# add users from validation/test-set iteratively to improve network
AL_USE_ADAPTIVE_LEARNING = True

# number of epoch from which on tweets from test-set are added
AL_NUM_EPOCHS_INCLUSION = 6

# percentage of users added to training-set per epoch
AL_PERCENTAGE_ADDED_PER_EPOCH = 0.1

# on each iteration not the top x of y users, but the top x/n users for all n groups are added
AL_EQUAL_CLASS_LEARNING = True

AL_LEARNING_RATE = 0.00001
AL_MOMENTUM = 0.9