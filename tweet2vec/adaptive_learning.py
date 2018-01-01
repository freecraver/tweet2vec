from __future__ import division
import numpy as np
import pandas as pd
import random
from tweet2vec.settings_char import DEBUG_MODE, AL_EQUAL_CLASS_LEARNING, GROUP_VALIDATION_MODE, GV_AVERAGE, GV_ARGMAX


def predict_user(single_predictions, users, n_classes, nr_top_value=None, labeldict=None):
    """
    Override single tweet-predictions by the predictions of their user
    :param single_predictions: matrix of probabilities (tweet X classes)
    :param users: list of authors for given tweets, has to be in same order as single_predictions
    :param n_classes: amount of different classes
    :param nr_top_value: amount users that are returned
    :return: Matrix of class-predictions (all tweets of a user have the same values);
            Boolean matrix for tweets, telling if they should be included to training-set
            (only when nr_top_value is not None)
    """

    np_single_predictions = np.array(single_predictions)
    unique_users, user_indices, user_cnt = np.unique(users, return_inverse=True, return_counts=True)
    avg_class_probabilities = np.zeros(len(unique_users) * n_classes).reshape(len(unique_users), n_classes)

    if GROUP_VALIDATION_MODE == GV_AVERAGE:
        # get average probability for each user as an array with a value for each class
        for i in range(len(unique_users)):
            avg_class_probabilities[i] = [sum(x) / user_cnt[i] for x in zip(*np_single_predictions[user_indices == i])]
    elif GROUP_VALIDATION_MODE == GV_ARGMAX:
        # get class-index for max-classification for each tweet
        np_single_argmax = np.argmax(np_single_predictions, axis=1)
        # set weights according to max-classified tweets
        for i in range(len(unique_users)):
            avg_class_probabilities[i] = np.bincount(np_single_argmax[user_indices == i],
                                                     minlength=n_classes).astype('float32') / user_cnt[i]
    else:
        raise Exception("Invalid validation mode (%s) - please check settings.char" % GROUP_VALIDATION_MODE)

    # order by descending probability
    rank = np.argsort(avg_class_probabilities)[:, ::-1]

    if nr_top_value is None:
        # default logic
        return rank[user_indices]
    else:
        # check which users should be included to training set
        included_user_indices = get_included_user_indices(avg_class_probabilities, nr_top_value)
        if DEBUG_MODE:
            pd_frame = pd.DataFrame(avg_class_probabilities, index=unique_users,
                                    columns=np.array([u'none'] + list(labeldict.keys())))
            print(pd_frame)
            print("---- users added to training set ----")
            print(pd_frame.loc[unique_users[included_user_indices]])

        included_user_matrix = np.isin(user_indices, included_user_indices)
        return rank[user_indices], included_user_matrix


def get_included_user_indices(avg_class_probabilities, nr_top_value):
    """
        depending on evaluation method this function returns an arrays of indices (of array unique_users)
        of users which should be included to training set
    :param avg_class_probabilities: average predictions per class
    :param nr_top_value: number of users, which should be included
    :return: indices of included users
    """
    if AL_EQUAL_CLASS_LEARNING:
        group_amount = int(nr_top_value/(len(avg_class_probabilities[0])-1))
        # indices for highest predictions of a given class (-> user indices)
        idx_max_class_prediction = np.argsort(avg_class_probabilities, axis=0)[:(group_amount*-1)-1:-1]
        # indices for highest predictions of a given user (-> class indices)
        idx_max_user_prediction = np.argmax(avg_class_probabilities, axis=1)
        included_user_indices = []
        for user_idx,  class_idx in enumerate(idx_max_user_prediction):
            if user_idx in idx_max_class_prediction[:, class_idx]:
                # user is one of the highest predicted users of this class
                included_user_indices.append(user_idx)
    else:
        # max value per row (how certain can a user be classified?)
        certainty = np.amax(avg_class_probabilities, axis=1)
        # include only top X users
        included_user_indices = np.argsort(certainty)[:(int(nr_top_value)*-1)-1:-1]

    return included_user_indices


def get_equal_subclasses(df):
    """
    :param df: pandas-dataframe with either 3 (mapped to [user, group, tweet]) or 2 (m. t. [group, tweet]) columns
    :return: pandas-dataframe in same format, containing the same amount of tweets for all classes
    """
    if len(df.columns) == 3:
        df.columns = ["user", "group", "tweet"]
    else:
        df.columns = ["group", "tweet"]

    groupby_obj = df.groupby("group")
    group_cnt = groupby_obj["tweet"].count()
    if len(df.columns) != 3:
        df_new = df.groupby("group").apply(lambda x: x.sample(group_cnt.min()))
    else:
        dfs = []
        # get the mean number of tweets per user for the group with the least amount of tweets
        # this is used to get a similar number of users per group
        mean_tweet_per_user = int(groupby_obj.get_group(group_cnt.idxmin()).groupby(["user"])["tweet"].count().mean())
        for group_name in groupby_obj.groups:
            df_group = groupby_obj.get_group(group_name).sample(frac=1)
            user_groupby_obj = df_group.groupby("user")
            tweets_left = group_cnt.min()
            start_idx = 0
            while tweets_left > 0:
                user_list = user_groupby_obj.groups.keys()
                random.shuffle(user_list)
                for user_name in user_list:
                    # try getting avg number of tweets
                    df_user = user_groupby_obj.get_group(user_name)[start_idx:
                                                                    start_idx + min(mean_tweet_per_user, tweets_left)]
                    dfs.append(df_user)
                    tweets_left -= df_user["tweet"].count()
                    if tweets_left <= 0:
                        break
                start_idx += mean_tweet_per_user

        df_new = pd.concat(dfs)

    return df_new


def get_equal_subclasses_arr(arr_user, arr_group, arr_tweet):
    """
    maps three numpy-arrays to a pandas-dataframe and calls subsampling function
    :return: three subsampled numpy arrays
    """
    # use tweet indices instead of tweets (to cope with multidimensionality)
    tweet_indices = np.arange(np.size(arr_tweet,0))
    cols = {"user":arr_user, "group": arr_group, "tweet": tweet_indices}
    df_tweets = pd.DataFrame(cols)
    # restore order (dict removes order)
    df_tweets = df_tweets[["user", "group", "tweet"]]
    df_reduced = get_equal_subclasses(df_tweets)
    return df_reduced["user"].as_matrix(), df_reduced["group"].as_matrix(), arr_tweet[df_reduced["tweet"],:]