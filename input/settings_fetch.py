# Define all global non-tweet2vec-specific params here

# File holding class labels
CLASS_LABELS_FILE = "input/classLables.txt"

# File holding list of users that were already fetched (used for split-processing)
ALREADY_FETCHED_USERS_FILE = "input/fetchedUsers.txt"

# File holding input for training, in format <label><TAB><Tweet>
TRAINING_FILE = "input/tweet_training.txt"

# File holding input for validation, in format <label><TAB><Tweet>
VALIDATION_FILE = "input/tweet_validation.txt"

# File used for specifying the twitter-users that should be fetched (csv of format <TwitterHandle>,<label>
USER_INPUT_FILE = "input/user.csv"

# Used for validation set, prints also username to file, uses VALIDATION_FILE instead of TRAINING_FILE
IS_VALIDATION = True

# Path where model-specific files are stored (word & label-dict,..)
PATH_MODEL = "model"

# Path where results should be stored
PATH_RESULT = "result/"

# Specifies, if the amount of Twitter-users per class should be equalized
EQUALIZE_USERS_FOR_CLASSES = True

# fill in your Twitter-handles
CONSUMER_KEY="b2PbrS2nGK4anadj7gtD9dFq5"
CONSUMER_SECRET="fvSZmshQRe8V0w7DnfNXxRkFvFWOHw7ScRb1q9Xw0AFEFLB7yq"
ACCESS_TOKEN="470209050-x9sI8xouIVsw2YSSjyOG6aHFDlRPIeVwJepjatAz"
ACCESS_TOKEN_SECRET="1Gi09PyW79NPmnjGZ9aM5e3kFeBqTSZT0FGKnCuxdfC8S"

# Tweet-properties
# Datetimes in the format 'DD.MM.YYYY hh:mm:ss'
START_TIME = "02.06.2017 00:00:00"
END_TIME = "01.08.2017 00:00:00"

# maximum number of tweets per user (leave empty if no max value should be used)
# NOTE: most recent (=newer) tweets are used before older tweets
MAX_TWEETS_PER_USER=300

# specifies if retweets of a user are used; if yes they are treated as if the user herself tweeted them
USE_RETWEETS = True