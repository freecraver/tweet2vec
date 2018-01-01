#!/usr/bin/env python
# encoding: utf-8

import tweepy  # https://github.com/tweepy/tweepy
import csv
import pandas as pd
import codecs
import random
from input.settings_fetch import USER_INPUT_FILE, ALREADY_FETCHED_USERS_FILE, TRAINING_FILE, IS_VALIDATION,\
    VALIDATION_FILE, USE_RETWEETS
from input.settings_fetch import ACCESS_TOKEN, ACCESS_TOKEN_SECRET, CONSUMER_KEY, CONSUMER_SECRET
from input.settings_fetch import EQUALIZATION_METHOD, START_TIME, END_TIME, MAX_TWEETS_PER_USER, MIN_TWEETS_PER_USER,\
    EQ_USER_FOR_CLASS, EQ_TWEETS_FOR_CLASS
from datetime import datetime

from collections import Counter

output_delimiter = "\t"

# this variable stores the id of the oldest tweet tw, for which tw.created_at > end_datetime
timespan_max_id = None


def get_all_tweets_for_user(username):
    """

    :reference: http://tweepy.readthedocs.io/en/v3.5.0/api.html#api-reference
    :param username: twitter-user to fetch
    :return: nothing
    """
    global timespan_max_id

    api = tweepy.API(auth)

    user_tweets = list()
    start_datetime = datetime.strptime(START_TIME, "%d.%m.%Y %H:%M:%S")
    end_datetime = datetime.strptime(END_TIME, "%d.%m.%Y %H:%M:%S")

    print("---------------------")
    print("fetching tweets for user %s before id %s" % (username, timespan_max_id or "[MAX]"))

    try:
        # fetch max cnt of tweets
        if timespan_max_id is None:
            fetched_tweets = api.user_timeline(id=username, count=200, tweet_mode='extended', include_rts=USE_RETWEETS)
        else:
            fetched_tweets = api.user_timeline(id=username, count=200, max_id=timespan_max_id, tweet_mode='extended',
                                               include_rts=USE_RETWEETS)

        # add tweets which are within the time-boundaries
        user_tweets.extend([tw for tw in fetched_tweets if start_datetime <= tw.created_at <= end_datetime])

        if len(fetched_tweets) < 1:
            print("No tweets in timespan found for user %s" % username)
            return user_tweets

        while MAX_TWEETS_PER_USER is None or len(user_tweets) < MAX_TWEETS_PER_USER:
            if fetched_tweets[-1].created_at > end_datetime:
                # set new upper bound for tweet-id
                timespan_max_id = min([timespan_max_id or float("inf"), fetched_tweets[-1].id])

            max_id = fetched_tweets[-1].id-1
            print("getting tweets before %s (%s)" % (max_id, fetched_tweets[-1].created_at))

            # fetch more tweets
            fetched_tweets = api.user_timeline(id=username, count=200, max_id=max_id, tweet_mode='extended',
                                               include_rts=USE_RETWEETS)

            user_tweets.extend([tw for tw in fetched_tweets if start_datetime <= tw.created_at <= end_datetime])

            if len(fetched_tweets) < 1:
                # no more tweets
                print("all %s tweets of %s fetched" % (len(user_tweets), username))
                break
            else:
                print("...%s tweets of %s added so far" % (min(len(user_tweets), MAX_TWEETS_PER_USER), username))
                if fetched_tweets[-1].created_at < start_datetime:
                    print("fetched last tweet in timespan for %s" % username)
                    break

        if MAX_TWEETS_PER_USER is None:
            return user_tweets
        else:
            return user_tweets[:MAX_TWEETS_PER_USER]

    except tweepy.TweepError:
        print("Could not fetch tweets for user %s, please check if the user exists or is protected" % username)


def get_fetched_users(file_path):
    """

    :param file_path: path to plaintext - file, containing usernames on each line
    :return: list of all already fetched users
    """

    with open(file_path) as f:
        return f.read().splitlines()


def get_user_with_category_from_csv(file_path):
    """

    :param file_path: path to csv-file in format <userhandle>;<category>
    :return: list containing [<userhandle>,<category>]
    """
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        cat_list = list(reader)

    return cat_list


def get_equal_user_subclasses(user_list):
    """
    :param user_list: list containing [<userhandle>,<category>]
    :return: list with the first n entries for each category,
                where n is the amount of entries for the least common category
    """
    counter = Counter(u[1] for u in user_list)
    min_cnt = counter.most_common()[-1][1]

    # reset counter
    for x in counter:
        counter[x] = 0

    # create a capped list, starting from the beginning
    capped_list = list()
    for u in user_list:
        if counter[u[1]] < min_cnt:
            capped_list.append(u)
            counter[u[1]] += 1

    return capped_list


def filter_equal_tweet_subclasses(file_name):
    """
    reads in all tweets from file and writes back equal number of tweets for all classes
    :param file_name: file containing tweets
    :return: nothing
    """
    print("Filtering tweets to obtain equal subclasses")
    df = pd.read_csv(file_name, sep='\t', header=None)
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
                                                                    start_idx+min(mean_tweet_per_user, tweets_left)]
                    dfs.append(df_user)
                    tweets_left -= df_user["tweet"].count()
                    if tweets_left <= 0:
                        break
                start_idx += mean_tweet_per_user

        df_new = pd.concat(dfs)

    df_new.to_csv(file_name, sep='\t', header=False, index=False)
    print("Reduced tweets from %d to %d" % (df["tweet"].count(), df_new["tweet"].count()))


if __name__ == "__main__":
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

    user_list = get_user_with_category_from_csv(USER_INPUT_FILE)

    output_file = VALIDATION_FILE if IS_VALIDATION else TRAINING_FILE

    if EQUALIZATION_METHOD == EQ_USER_FOR_CLASS:
        user_list = get_equal_user_subclasses(user_list)

    processed_users = get_fetched_users(ALREADY_FETCHED_USERS_FILE)

    for entry in user_list:
        if not entry[0] in processed_users:
            user_tweets = get_all_tweets_for_user(entry[0])
            if user_tweets is None:
                # couldn't fetch tweets
                continue

            if MIN_TWEETS_PER_USER is None or MIN_TWEETS_PER_USER <= len(user_tweets):
                # write all tweets with preceding category and delimiter
                with codecs.open(output_file, 'a', 'utf8') as f:
                    for tweet in user_tweets:
                        tweet_text = ""
                        # write <original tweet> instead of RT @userabc:<original tweet>
                        if hasattr(tweet, "retweeted_status"):
                            tweet_text = tweet.retweeted_status.full_text
                        else:
                            tweet_text = tweet.full_text

                        tweet_text = tweet_text.replace("\n", " ")
                        if not IS_VALIDATION:
                            f.write("%s%s%s\n" % (entry[1], output_delimiter, tweet_text))
                        else:
                            # also include username
                            f.write("%s%s%s%s%s\n" % (entry[0], output_delimiter, entry[1], output_delimiter, tweet_text))
            else:
                print("removed user %s, because only %d tweets were found" % (entry[0], len(user_tweets)))

            # write user to fetched_user_file, if fetching process is terminated
            with open(ALREADY_FETCHED_USERS_FILE, 'a') as f:
                f.write(entry[0]+"\n")
                processed_users.append(entry[0])

    if EQUALIZATION_METHOD == EQ_TWEETS_FOR_CLASS:
        filter_equal_tweet_subclasses(output_file)
