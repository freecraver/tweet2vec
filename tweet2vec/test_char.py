import numpy as np
import lasagne
import theano
import theano.tensor as T
import sys
import os
import tweet2vec.batch_char as batch
import pickle as pkl
import io
import tweet2vec.evaluate
import tweet2vec.adaptive_learning as adaptive_learning
from math import ceil

from tweet2vec.t2v import tweet2vec, init_params, load_params
from tweet2vec.settings_char import N_BATCH, MAX_LENGTH, MAX_CLASSES, REGULARIZATION, IS_GROUPED_VALIDATION, \
    AL_USE_ADAPTIVE_LEARNING, AL_PERCENTAGE_ADDED_PER_EPOCH, AL_LEARNING_RATE, AL_MOMENTUM


def classify(tweet, t_mask, params, n_classes, n_chars):
    # tweet embedding
    emb_layer = tweet2vec(tweet, t_mask, params, n_chars)
    # Dense layer for classes
    l_dense = lasagne.layers.DenseLayer(emb_layer, n_classes, W=params['W_cl'], b=params['b_cl'],
                                        nonlinearity=lasagne.nonlinearities.softmax)

    return lasagne.layers.get_output(l_dense), l_dense, lasagne.layers.get_output(emb_layer)


def load_network(model_path, n_classes, n_char, m_num=None,):
    print("Loading model params...")
    if m_num is None:
        l_params = load_params('%s/best_model.npz' % model_path)
    else:
        l_params = load_params('%s/model_%d.npz' % (model_path, m_num))

    print("Building network...")
    # Tweet variables
    l_tweet = T.itensor3()
    l_targets = T.ivector()

    # masks
    l_t_mask = T.fmatrix()

    # network for prediction
    predictions, net, embeddings = classify(l_tweet, l_t_mask, l_params, n_classes, n_char)

    if AL_USE_ADAPTIVE_LEARNING:
        loss = lasagne.objectives.categorical_crossentropy(predictions, l_targets)
        cost = T.mean(loss) + REGULARIZATION *\
               lasagne.regularization.regularize_network_params(net, lasagne.regularization.l2)

        print("Computing updates...")
        lr = AL_LEARNING_RATE
        mu = AL_MOMENTUM
        updates = lasagne.updates.nesterov_momentum(cost, lasagne.layers.get_all_params(net), lr, momentum=mu)
        inps = [l_tweet, l_t_mask, l_targets]
        l_train = theano.function(inps, cost, updates=updates)

    # Theano function
    print("Compiling theano functions...")
    l_predict = theano.function([l_tweet, l_t_mask], predictions)
    l_encode = theano.function([l_tweet, l_t_mask], embeddings)

    if AL_USE_ADAPTIVE_LEARNING:
        return l_predict, l_encode, l_train
    else:
        return l_predict, l_encode, None


def train_adaptively(train_function, new_tweets, new_targets):
    print("Training...")
    i32_targets = new_targets.astype('int32')
    indices = np.random.permutation(np.arange(len(new_tweets)))
    for i in range(int(ceil(float(len(new_tweets))/N_BATCH))):
        start_idx = i*N_BATCH
        stop_idx = min(start_idx+N_BATCH,len(new_tweets))
        cur_indices = indices[start_idx:stop_idx]
        mask = np.ones((stop_idx-start_idx,MAX_LENGTH)).astype('float32')
        # updates network via shared variables
        train_function(new_tweets[cur_indices], mask, i32_targets[cur_indices])


def main(args):
    data_path = args[0]
    model_path = args[1]
    save_path = args[2]
    m_num = int(args[3]) if len(args) > 3 else None

    print("Preparing Data...")
    # Test data
    Xt = []
    yt = []
    userT = []

    with io.open(data_path, 'r', encoding='utf-8') as f:
        if IS_GROUPED_VALIDATION:
            for line in f:
                l_res = line.rstrip('\n').split('\t')
                if len(l_res) > 2:
                    (userC, yc, Xc) = l_res[0], l_res[1], l_res[2]
                    Xt.append(Xc[:MAX_LENGTH])
                    yt.append(yc.split(','))
                    userT.append(userC)
                else:
                    print("skipped line %s" % line)
        else:
            for line in f:
                # TODO: remove _
                (_, yc, Xc) = line.rstrip('\n').split('\t')
                Xt.append(Xc[:MAX_LENGTH])
                yt.append(yc.split(','))

    print("Loading dictionaries...")
    with open('%s/dict.pkl' % model_path, 'rb') as f:
        chardict = pkl.load(f)
    with open('%s/label_dict.pkl' % model_path, 'rb') as f:
        labeldict = pkl.load(f)
    n_char = len(chardict.keys()) + 1
    n_classes = min(len(labeldict.keys()) + 1, MAX_CLASSES)

    predict, encode, _ = load_network(model_path, n_classes, n_char, m_num)

    # iterators
    if IS_GROUPED_VALIDATION:
        test_iter = batch.BatchTweets(Xt, yt, labeldict, batch_size=N_BATCH, max_classes=MAX_CLASSES, test=True,
                                      users=userT)
    else:
        test_iter = batch.BatchTweets(Xt, yt, labeldict, batch_size=N_BATCH, max_classes=MAX_CLASSES, test=True)


    # Test
    print("Testing...")

    iteration_nr = 1

    while True:
        out_data = []
        out_pred = []
        out_emb = []
        out_target = []
        if AL_USE_ADAPTIVE_LEARNING:
            print("%d. Iteration..." % iteration_nr)
        if IS_GROUPED_VALIDATION:
            # Store predictions and users to be able to override them before validation
            single_preds = []
            single_users = []
            tweets = []

            for xr, y, users in test_iter:
                x, x_m = batch.prepare_data(xr, chardict, n_chars=n_char)
                p = predict(x, x_m)
                e = encode(x, x_m)
                single_preds.extend(p)
                single_users.extend(users)
                out_target.extend(y)
                out_data.extend(xr)
                out_emb.append(e)
                if AL_USE_ADAPTIVE_LEARNING:
                    tweets.extend(x)

            out_pred = adaptive_learning.predict_user(single_preds, single_users, n_classes, labeldict=labeldict)
        else:
            for xr, y in test_iter:
                x, x_m = batch.prepare_data(xr, chardict, n_chars=n_char)
                p = predict(x, x_m)
                e = encode(x, x_m)
                ranks = np.argsort(p)[:, ::-1]

                for idx, item in enumerate(xr):
                    out_data.append(item)
                    out_pred.append(ranks[idx, :])
                    out_emb.append(e[idx, :])
                    out_target.append(y[idx])

        if AL_USE_ADAPTIVE_LEARNING:
            nr_included_users = len(set(single_users)) * AL_PERCENTAGE_ADDED_PER_EPOCH * iteration_nr
            preds, tweet_included = adaptive_learning.predict_user(single_preds, single_users,
                                                                   n_classes, nr_included_users, labeldict)
            new_tweets = np.array(tweets)[tweet_included]
            new_targets = np.array(preds)[tweet_included][:, 0]

            _, new_targets, new_tweets = adaptive_learning.get_equal_subclasses_arr(
                np.array(single_users)[tweet_included],
                new_targets,
                new_tweets)

            if len(new_tweets) > 0:
                # reset network to overthrow current modifications
                predict, encode, train = load_network(model_path, n_classes, n_char, m_num)

                # learn classifications were model was most confident
                train_adaptively(train, new_tweets, new_targets)

            # store model
            save_path = args[2] + "/" + str(iteration_nr) + ". Iteration"
            iteration_nr += 1

        # create dirs for iteration step
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Save
        print("Saving...")
        with open('%s/data.pkl' % save_path, 'wb') as f:
            pkl.dump(out_data, f)
        with open('%s/predictions.npy' % save_path, 'wb') as f:
            np.save(f, np.asarray(out_pred))
        with open('%s/embeddings.npy' % save_path, 'wb') as f:
            np.save(f, np.asarray(out_emb))
        with open('%s/targets.pkl' % save_path, 'wb') as f:
            pkl.dump(out_target, f)
        with open('%s/users.pkl' % save_path, 'wb') as f:
            pkl.dump(single_users, f)

        if not AL_USE_ADAPTIVE_LEARNING or (AL_PERCENTAGE_ADDED_PER_EPOCH * (iteration_nr-1)) >= 1:
            break

if __name__ == '__main__':
    main(sys.argv[1:])
    evaluate.main(sys.argv[3], sys.argv[2])
