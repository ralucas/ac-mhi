import numpy as np
import unittest
import acmhi
import os
import cv2

INPUT_DIR = "input_images/input_test"

ACTIONS = [
    "boxing",
    "handclapping",
    "handwaving",
    "jogging",
    "running",
    "walking"
]

class AC_Test(unittest.TestCase):

    def test_binary_image(self):
        ac = acmhi.ActivityQuantifier("walking", "person01_walking_d1_uncomp.avi")
        video = ac.load_video("walking", "person01_walking_d1_uncomp.avi")
        c = 0
        curr_img = ac.prepare_frame(video.next())
        while c < 25:
            prev_img = curr_img
            curr_img = ac.prepare_frame(video.next())
            c += 1

        result = ac.binary_mei(curr_img, prev_img)
        cv2.imwrite("test_output/binary_img.png", result * 255)
        cv2.imwrite("test_output/binary_img_curr.png", curr_img)
        self.assertTrue(np.any(result), "Binary MEI Image")

    def test_mei(self):
        ac = acmhi.ActivityQuantifier(np.random.choice(ACTIONS))
        meis, _ = ac.build_mei_mhi()
        c = 0
        for mei in meis:
            mei_image = mei * 255
            cv2.imwrite(
                "test_output/mei_image_{}_person{}_d{}_{}.png".format(
                    ac.action,
                    ac.person_num,
                    ac.d_num,
                    c
                ),
                mei_image
            )
            c += 1

            # MEI image should be neither all white or black
            self.assertFalse(np.all(mei_image == 255), "MEI is incorrect, all white")
            self.assertFalse(np.all(mei_image == 0), "MEI is incorrect, all black")

    def test_mhi(self):
        ac = acmhi.ActivityQuantifier(np.random.choice(ACTIONS))
        _, mhis = ac.build_mei_mhi()
        c = 0
        for mhi in mhis:
            mhi_image = np.int32(np.around((np.float32(mhi) / ac.tau) * 255.))
            cv2.imwrite(
                "test_output/mhi_image_{}_person{}_d{}_{}.png".format(
                    ac.action,
                    ac.person_num,
                    ac.d_num,
                    c
                ),
                mhi_image
            )
            c += 1
            # MEI image should be neither all white or black
            gtz = mhi_image[mhi_image > 0]
            test = gtz[gtz < 255]
            self.assertTrue(np.any(test), "MHI is incorrect, b&w")

    def test_moments(self):
        ac = acmhi.ActivityQuantifier(np.random.choice(ACTIONS))
        meis, mhis = ac.build_mei_mhi()
        for mhi in mhis:
            M_00, M_10, M_01 = ac.img_moments(mhi)
            moments = cv2.moments(np.uint8(mhi))
            self.assertTrue(np.isclose(moments['m00'], M_00, atol=3.),
                "{} != {}".format(moments['m00'], M_00)
            )
            self.assertTrue(np.isclose(moments['m10'], M_10, atol=3.),
                "{} != {}".format(moments['m10'], M_10)
            )
            self.assertTrue(np.isclose(moments['m01'], M_01, atol=3.),
                "{} != {}".format(moments['m01'], M_01)
            )

    def test_central_moments(self):
        ac = acmhi.ActivityQuantifier(np.random.choice(ACTIONS))
        meis, mhis = ac.build_mei_mhi()
        for mhi in mhis:
            mus, vs = ac.central_moments(mhi)
            c_moms = cv2.moments(np.uint8(mhi))
            self.assertTrue(np.isclose(mus['20'], c_moms['mu20'], rtol=0.01),
                            "{} != {}".format(mus['20'], c_moms['mu20']))
            self.assertTrue(np.isclose(mus['11'], c_moms['mu11'], rtol=0.01),
                            "{} != {}".format(mus['11'], c_moms['mu11']))
            self.assertTrue(np.isclose(mus['12'], c_moms['mu12'], rtol=0.01),
                            "{} != {}".format(mus['12'], c_moms['mu12']))
            self.assertTrue(np.isclose(mus['02'], c_moms['mu02'], rtol=0.1),
                            "{} != {}".format(mus['02'], c_moms['nu02']))
            self.assertTrue(np.isclose(mus['03'], c_moms['mu03'], rtol=0.1),
                            "{} != {}".format(mus['03'], c_moms['nu03']))
            self.assertTrue(np.isclose(mus['21'], c_moms['mu21'], rtol=0.1),
                            "{} != {}".format(mus['21'], c_moms['mu21']))
            self.assertTrue(np.isclose(mus['30'], c_moms['mu30'], rtol=0.1),
                            "{} != {}".format(mus['30'], c_moms['mu30']))

    def test_scale_invariant_moments(self):
        ac = acmhi.ActivityQuantifier(np.random.choice(ACTIONS))
        meis, mhis = ac.build_mei_mhi()
        for mhi in mhis:
            mus, vs = ac.central_moments(mhi)
            c_moms = cv2.moments(np.uint8(mhi))
            self.assertTrue(np.isclose(vs['20'], c_moms['nu20'], rtol=0.1),
                            "{} != {}".format(vs['20'], c_moms['nu20']))
            self.assertTrue(np.isclose(vs['11'], c_moms['nu11'], rtol=0.1),
                            "{} != {}".format(vs['11'], c_moms['nu11']))
            self.assertTrue(np.isclose(vs['12'], c_moms['nu12'], rtol=0.1),
                            "{} != {}".format(vs['12'], c_moms['nu12']))
            self.assertTrue(np.isclose(vs['02'], c_moms['nu02'], rtol=0.1),
                            "{} != {}".format(vs['02'], c_moms['nu02']))
            self.assertTrue(np.isclose(vs['03'], c_moms['nu03'], rtol=0.1),
                            "{} != {}".format(vs['03'], c_moms['nu03']))
            self.assertTrue(np.isclose(vs['21'], c_moms['nu21'], rtol=0.1),
                            "{} != {}".format(vs['21'], c_moms['nu21']))
            self.assertTrue(np.isclose(vs['30'], c_moms['nu30'], rtol=0.1),
                            "{} != {}".format(vs['30'], c_moms['nu30']))

    def test_hu_moments(self):
        for _ in range(100):
            ac = acmhi.ActivityQuantifier(np.random.choice(ACTIONS))
            meis, mhis = ac.build_mei_mhi()
            for mhi in mhis:
                hus = ac.hu_moments(mhi)
                c_moms = cv2.moments(np.uint8(mhi))
                compare = cv2.HuMoments(c_moms)
                for i, hu in enumerate(hus):
                    self.assertTrue(np.isclose(hu, compare[i][0], rtol=0.01),
                        "{} != {} :: {}".format(hu, compare[i][0], i+1))

class AT_Test(unittest.TestCase):

    def test_split(self):
        at = acmhi.ActivityTrainer(dir='input_videos', split_percent=0.5)
        train_set, test_set = at.split_dataset()
        self.assertTrue(train_set.keys() == test_set.keys())
        for k, v in train_set.items():
            self.assertTrue(len(v))
            self.assertTrue(len(test_set[k]))

    def test_train(self):
        at = acmhi.ActivityTrainer(dir='input_videos', split_percent=0.5)
        at.train()
        pred_train, pred_test = at.test_predict()
        train_percent = 100 * (np.sum(pred_train == at.ytrain, dtype=np.float64) / np.float64(len(at.ytrain)))
        test_percent = 100 * (np.sum(pred_test == at.ytest, dtype=np.float64) / np.float64(len(at.ytest)))
        print "Train {}% {} expected {}".format(
            train_percent,
            str(pred_train[pred_train != at.ytrain]),
            str(at.ytrain[at.ytrain != pred_train])
        )
        print "Test {}% {} expected {}".format(
            test_percent,
            str(pred_test[pred_test != at.ytest]),
            str(at.ytest[at.ytest != pred_test])
        )
        self.assertTrue(train_percent > 80.)
        self.assertTrue(test_percent > 70.)

    def test_train_actions(self):
        at = acmhi.ActivityTrainer(
            dir='input_videos',
            split_percent=0.7,
            # actions=["handclapping", "boxing", "walking", "jogging", "running"],
            trainer="knn"
        )
        at.train()
        pred_train, pred_test = at.test_predict()
        pred_train = np.array(pred_train)
        pred_test = np.array(pred_test)
        at.ytrain = np.array(at.ytrain)
        at.ytest = np.array(at.ytest)
        train_percent = 100 * (np.sum(pred_train == at.ytrain, dtype=np.float64) / np.float64(len(at.ytrain)))
        test_percent = 100 * (np.sum(pred_test == at.ytest, dtype=np.float64) / np.float64(len(at.ytest)))
        vec = np.vectorize(lambda x: at.labels[x])
        unique, counts = np.unique(vec(pred_train[pred_train != at.ytrain]), return_counts=True)
        offs = dict(zip(unique, counts))
        print "Train {}%\n {} \nexpected \n{}\n{}".format(
            train_percent,
            str(vec(pred_train[pred_train != at.ytrain])),
            str(vec(at.ytrain[at.ytrain != pred_train])),
            str(offs)
        )
        unique, counts = np.unique(vec(pred_test[pred_test != at.ytest]), return_counts=True)
        offs = dict(zip(unique, counts))
        print "Test {}%\n {} \nexpected \n{}\n{}".format(
            test_percent,
            str(vec(pred_test[pred_test != at.ytest])),
            str(vec(at.ytest[at.ytest != pred_test])),
            str(offs)
        )
        self.assertTrue(train_percent > 80., "Failed Training {}".format(train_percent))
        self.assertTrue(test_percent > 80., "Failed Test {}".format(test_percent))

if __name__ == '__main__':
    unittest.main()
