import cv2
import os
import datetime
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import json

import acmhi

ACTIONS = [
    "boxing",
    "handclapping",
    "handwaving",
    "jogging",
    "running",
    "walking"
]

def sampler():
    action = ACTIONS[np.random.randint(0,6)]
    return "person{}_{}_d{}_uncomp.avi".format(
        str(np.random.randint(1,26)).zfill(2),
        action,
        np.random.randint(1,5)
    )

def get_video_filename(action, person_num, d_num):
    filename = "person{}_{}_d{}_uncomp.avi".format(
        str(person_num).zfill(2),
        action,
        d_num
    )
    return filename


# Taken from sklearn docs:
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        # cm[cm == 0.] = 1.
        cm = 100 * (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])
        # cm[cm == ((1./3.)*100)] = 0.
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def exp_mei():
    for i in range(20):
        file = get_video_filename("walking", i+1, np.random.randint(1,5))
        ac = acmhi.ActivityQuantifier("walking", filename=file)
        mei, _ = ac.build_mei_mhi()
        mei_image = np.int32(mei * 255.)
        cv2.imwrite(
            "output/mei_image_{}_person{}_d{}.png".format(
                ac.action,
                ac.person_num,
                ac.d_num
            ),
            mei_image
        )

def exp_mei_mhi(act="handclapping"):
    for i in range(5):
        print "Running mei and mhi images for " + act
        file = get_video_filename(act, i+1, 3)#np.random.randint(1,5))
        ac = acmhi.ActivityQuantifier(act, filename=file)
        meis, mhis = ac.build_mei_mhi()
        for i, mhi in enumerate(mhis):
            mhi_image = np.int32(np.around((np.float32(mhi) / ac.tau) * 255.))
            mei_image = np.int32(meis[i] * 255.)
            cv2.imwrite(
                "output/mhi_image_{}_person{}_d{}_{}.png".format(
                    ac.action,
                    ac.person_num,
                    ac.d_num,
                    i
                ),
                mhi_image
            )
            cv2.imwrite(
                "output/mei_image_{}_person{}_d{}_{}.png".format(
                    ac.action,
                    ac.person_num,
                    ac.d_num,
                    i
                ),
                mei_image
            )

def conf_matrix():
    at = acmhi.ActivityTrainer(
        dir='input_videos',
        split_percent=0.7,
        trainer="knn",
        create_pkl=False
    )
    at.train()
    pred_train, pred_test = at.test_predict()
    at.labels = np.array(at.labels)
    pred_train = at.labels[np.array(pred_train)]
    pred_test = at.labels[np.array(pred_test)]
    at.ytrain = at.labels[np.array(at.ytrain)]
    at.ytest = at.labels[np.array(at.ytest)]
    train_percent = np.around(100 * (np.sum(pred_train == at.ytrain, dtype=np.float64) / np.float64(len(at.ytrain))), decimals=1)
    test_percent = np.around(100 * (np.sum(pred_test == at.ytest, dtype=np.float64) / np.float64(len(at.ytest))), decimals=1)
    conf_train = confusion_matrix(at.ytrain, pred_train, labels=at.labels)
    conf_test = confusion_matrix(at.ytest, pred_test, labels=at.labels)
    print "Training: {}%".format(train_percent)
    print conf_train
    plt.figure()
    plot_confusion_matrix(conf_train, classes=at.labels, normalize=True,
          title="Training Confusion matrix | {}% correct".format(train_percent))
    plt.savefig("output/training_{}.png".format(datetime.datetime.now().isoformat()))
    print "Test: {}%".format(test_percent)
    print conf_test
    plt.figure()
    plot_confusion_matrix(conf_test, classes=at.labels, normalize=True,
          title="Testing Confusion matrix | {}% correct".format(test_percent))
    plt.savefig("output/test_{}.png".format(datetime.datetime.now().isoformat()))

def build_expectations(exps, labels):
    output = []
    for x in exps:
        for k, v in x.items():
            [beg, end] = k.split("-")
            i = int(beg)
            while i <= int(end):
                output.append(labels.index(v))
                i += 1
    return np.array(output)

def exp_video_activities():
    with open('experiment_videos.json') as json_data:
        exps = json.load(json_data)
    exp_videos = os.listdir("experiment_videos")
    for i, video_file in enumerate(exp_videos):
        ap = acmhi.ActivityPredictor(
            video_file=os.path.join("experiment_videos/", video_file),
            out_name="experiment_{}".format(i+1)
        )
        labls = list(ap.labels)
        labls.append('no_activity')
        expects = build_expectations(exps[video_file], labels=labls)
        new_video_file, preds = ap.predict()
        np_preds = np.array(preds)
        if len(np_preds) == len(expects):
            correct = np.around(100 * (np.sum(np_preds == expects, dtype=np.float64) / np.float64(len(expects))), decimals=1)
            plt.figure()
            conf_mat = confusion_matrix(expects, np_preds)
            conf_mat[conf_mat == np.nan] = 0.
            vname = video_file.split(".")[0]
            plot_confusion_matrix(conf_mat, classes=labls, normalize=True,
                                  title="{} confusion matrix | {}% correct".format(vname, correct))
            plt.savefig("output/{}_matrix.png".format(video_file))
        else:
            correct = "unknown"
            print "preds: {} != expects {}".format(len(np_preds), len(expects))
        print "======================================"
        print "Video created: " + str(new_video_file)
        print "Correct results: " + str(correct) + "%"
        print "======================================"

if __name__ == "__main__":
    for action in ACTIONS:
        exp_mei_mhi(action)
    conf_matrix()
    exp_video_activities()
