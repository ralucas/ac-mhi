import cv2
import os
import numpy as np
import json
import sklearn
import pickle
from sklearn import neighbors, svm, neural_network
import time
import operator

DEBUG = False

with open('sequences.json') as json_data:
    sequences = json.load(json_data)


class ActivityQuantifier:

    def __init__(self, action="", filename="", theta=21, tau=21):
        self.action = action
        self.filename = filename
        self.image_gen = self.load_video(action, filename)
        self.skip = 1
        self.theta = 14.
        self.taus = {
            'boxing': 7.,
            'handclapping': 13.,
            'handwaving': 17.,
            'jogging': 21.,
            'running': 11.,
            'walking': 35.
        }
        self.tau = np.float32(tau)

        if action != "":
            self.tau = self.taus[action]
        else:
            self.tau = np.around(np.mean(self.taus.values()))
        self.sequences = []
        self.total_frames = 0
        self.frame_count = 0
        self.fps = 30
        self.frame_size = (0, 0)

    def load_video(self, action="", filename=""):
        video_file = filename
        if not filename:
            filename = "person{}_{}_d{}_uncomp.avi".format(
                str(np.random.randint(1,26)).zfill(2),
                action,
                np.random.randint(1,5)
            )

        if action != "":
            s = filename.split("_")
            self.person_num = s[0].split("person")[1]
            self.d_num = s[2].split("d")[1]
            self.seq_name = "person{}_{}_d{}".format(
                self.person_num,
                action,
                self.d_num
            )
            self.sequences = sequences[self.seq_name]
            video_file = os.path.join("input_videos", action, filename)
        #print "Running " + str(video_file)
        video = cv2.VideoCapture(video_file)
        self.total_frames = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        self.frame_count = 0
        self.fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        self.frame_size = (video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT), video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        while video.isOpened():
            ret, frame = video.read()
            if ret:
                yield frame
            else:
                break
            self.frame_count += 1

        video.release()
        yield None


    def prepare_frame(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (13, 13), 1)
        morph_kern = np.ones((9, 9), dtype=np.int32)
        morph_frame = cv2.morphologyEx(blurred_frame, cv2.MORPH_OPEN, morph_kern)
        return np.array(morph_frame, dtype=np.int32)

    def binary_mei(self, curr_img, prev_img):
        mei_img = np.zeros(curr_img.shape, dtype=np.int32)
        img_diff = np.abs(np.subtract(curr_img, prev_img))
        mei_img[img_diff > self.theta] = 1
        return mei_img

    def mhi(self, mei_frame, mhi_img):
        prev_mhi = np.copy(mhi_img)
        mhi_img[mei_frame == 1] = self.tau
        sub = np.subtract(prev_mhi[mei_frame != 1], 1)
        sub[sub < 0] = 0
        mhi_img[mei_frame != 1] = sub
        return mhi_img

    def img_moments(self, img):
        y_len, x_len = img.shape
        x = np.arange(x_len)
        y = np.arange(y_len)
        M_00 = np.float32(np.sum(img))
        M_10 = np.float32(np.sum(x * np.sum(img, axis=0)))
        M_01 = np.float32(np.sum(y * np.sum(img, axis=1)))

        return M_00, M_10, M_01

    def central_moments(self, img):
        pq = np.array([[2,0], [1,1], [0,2], [3,0], [2,1], [1,2], [0,3]])#, [2,2]])
        p = pq[:,0]
        q = pq[:,1]
        M_00, M_10, M_01 = self.img_moments(img)
        if M_00 == 0 or M_00 == np.nan:
            print M_00
        x_bar = M_10 / M_00
        y_bar = M_01 / M_00
        y_len, x_len = img.shape
        x = np.arange(x_len)
        y = np.arange(y_len)
        xp = np.power(x - x_bar, p[:,np.newaxis], dtype=np.float32)
        yq = np.power(y - y_bar, q[:,np.newaxis], dtype=np.float32)
        xpi = xp * img[:,np.newaxis]
        yqi = yq.T
        xypq = xpi * yqi[:,:, np.newaxis]

        mu_pq = np.sum(np.sum(xypq, axis=0), axis=1)

        pow = 1 + np.divide(np.sum(pq, axis=1, dtype=np.float32), 2.)
        denom = np.power(M_00, pow)
        v_pq = mu_pq / denom

        cent_moms = {}
        si_moms = {}
        for i, pqs in enumerate(pq):
            cent_moms[str(pqs[0]) + str(pqs[1])] = mu_pq[i]
            si_moms[str(pqs[0]) + str(pqs[1])] = v_pq[i]

        return cent_moms, si_moms

    def hu_moments(self, img):
        _, nu = self.central_moments(img)
        hus = np.array([
            #1
            nu['20'] + nu['02'],
            #2
            np.square(nu['20'] - nu['02']) + (4 * np.square(nu['11'])),
            #3
            np.square(nu['30'] - (3*nu['12'])) + np.square((3*nu['21']) - nu['03']),
            #4
            np.square(nu['30'] + nu['12']) + np.square(nu['21'] + nu['03']),
            #5
            ((nu['30'] - (3*nu['12']))*(nu['30'] + nu['12'])*(
                (np.square(nu['30'] + nu['12'])) - (3 * (np.square(nu['21'] + nu['03'])))
            )) + ((3*nu['21'] - nu['03'])*(nu['21'] + nu['03']))*(
                (3*(np.square(nu['30'] + nu['12'])) - (np.square(nu['21'] + nu['03'])))
            ),
            #6
            ((nu['20'] - nu['02'])*(np.square(nu['30']+nu['12']) - np.square(nu['21'] + nu['03']))) +
                4*nu['11']*(nu['30'] + nu['12'])*(nu['21'] + nu['03'])
            #7
            # (((3*nu['21']) - nu['03'])*(nu['21'] + nu['03'])*(
            #     (3*(np.square(nu['30'] + nu['12']))) - np.square(nu['21'] + nu['03'])
            # )) - ((nu['30'] - (3*nu['12']))*(nu['21'] + nu['03'])*(
            #     (3*(np.square(nu['30'] + nu['12']))) - np.square(nu['21'] + nu['03'])
            # ))

            # (((3*nu['21']) - nu['03'])*(nu['30'] + nu['12'])*(
            #     np.square(nu['30'] + nu['12']) - (3*(np.square(nu['21'] + nu['03'])))
            # )) - ((-nu['30'] + (3*nu['12']))*(nu['21'] + nu['30'])*(
            #     (3*(np.square(nu['30'] + nu['12']))) - np.square(nu['21'] + nu['03'])
            # ))
        ])
        o = np.concatenate((hus, np.array(nu.values())))
        return o

    def jump_to_frame(self, frame, curr_count):
        if frame == 1:
            return self.image_gen
        while curr_count <= frame:
            self.image_gen.next()
            curr_count += 1
        return curr_count

    def build_mei_mhi(self):
        meis = []
        mhis = []
        count = 1
        prev_frame = self.prepare_frame(self.image_gen.next())

        sequences = self.sequences
        if len(self.sequences) == 0:
            sequences = [[1, self.total_frames]]

        seqs = []
        for sequence in sequences:
            num_seqs = (sequence[1] - sequence[0]) / self.tau
            for n in range(int(num_seqs)):
                start = sequence[0] + (self.tau * n)
                end = start + self.tau
                seqs.append([int(start), int(end)])

        for seq in seqs:
            mei_agg = np.zeros(prev_frame.shape, dtype=np.int32)
            mhi_image = np.zeros(prev_frame.shape, dtype=np.int32)

            if count != seq[0]:
                count = self.jump_to_frame(seq[0], count)
            while count <= seq[1]:
                frame = self.image_gen.next()
                if frame is None:
                    break
                for _ in range(self.skip-1):
                    frame = self.image_gen.next()
                    count += 1
                    if frame is None:
                        break
                if frame is None:
                    break

                morph_frame = self.prepare_frame(frame)
                if DEBUG and count == 25:
                    cv2.imwrite("test_output/morph"+str(count)+".png", morph_frame)

                mei_frame = self.binary_mei(morph_frame, prev_frame)

                if DEBUG and count == 25:
                    cv2.imwrite("test_output/mei_frame"+str(count)+".png", mei_frame*255)

                mhi_image = self.mhi(mei_frame, mhi_image)

                mei_agg += mei_frame

                if DEBUG:
                    mei_agg[mei_agg > 1] = 1
                    cv2.imwrite("test_output/{}_mei_agg.png".format(self.action), mei_agg*255)
                    cv2.imwrite("test_output/{}_curr_frame.png".format(self.action), morph_frame)

                prev_frame = morph_frame
                count += 1

            if (np.sum(mei_agg) == 0.):
                continue
            if (np.sum(mhi_image) == 0.):
                continue
            mei_agg[mei_agg > 1] = 1
            meis.append(mei_agg)
            mhis.append(mhi_image)
        return meis, mhis


class ActivityTrainer:
    def __init__(self, dir, split_percent=0.5, actions=[], trainer="knn", pkl_file=None, tau=21, theta=21, create_pkl=False):
        self.dir = dir
        self.actions = actions
        self.split_percent = split_percent
        self.training_set = {}
        self.test_set = {}
        self.split_dataset()
        self.labels = [
            'boxing',
            'handclapping',
            'handwaving',
            'jogging',
            'running',
            'walking'
        ]
        self.Xtrain = []
        self.ytrain = []
        self.Xtest = []
        self.ytest = []
        self.trainer_name = trainer
        self.pkl_file = pkl_file
        self.tau = tau
        self.theta = theta
        self.create_pkl = create_pkl

        if pkl_file is not None:
            trainer_pkl = open(pkl_file, "rb")
            self.trainer = pickle.load(trainer_pkl)
        elif trainer == "knn":
            self.trainer = neighbors.KNeighborsClassifier()
        elif trainer == "svm":
            self.trainer = svm.SVC(kernel="linear")
        elif trainer == "nn":
            self.trainer = neural_network.MLPClassifier()

    def split_dataset(self):
        videos = {}
        for (path, dirnames, filenames) in os.walk(self.dir):
            if len(dirnames) == 0:
                label = path.split('/')[1]
                if len(self.actions) and label in self.actions:
                    videos[label] = filenames
                elif len(self.actions) == 0:
                    videos[label] = filenames
        for k, v in videos.items():
            v = np.array(v)
            train_qty = int(len(v) * self.split_percent)
            np.random.shuffle(v)
            s = np.copy(v)
            self.training_set[k] = s[0:train_qty]
            self.test_set[k] = s[train_qty:len(s)]
        return self.training_set, self.test_set

    def build_training_sets(self):
        for k, vids in self.training_set.items():
            y_val = self.labels.index(k)
            for vid in vids:
                ac = ActivityQuantifier(k, vid, tau=self.tau, theta=self.theta)
                _, mhis = ac.build_mei_mhi()
                for mhi in mhis:
                    hus = ac.hu_moments(mhi)
                    self.Xtrain.append(hus)
                    self.ytrain.append(y_val)

    def build_test_sets(self):
        for k, vids in self.test_set.items():
            y_val = self.labels.index(k)
            for vid in vids:
                ac = ActivityQuantifier(k, vid, tau=self.tau, theta=self.theta)
                _, mhis = ac.build_mei_mhi()
                for mhi in mhis:
                    hus = ac.hu_moments(mhi)
                    self.Xtest.append(hus)
                    self.ytest.append(y_val)

    def train(self):
        if self.pkl_file is None:
            self.build_training_sets()
            self.trainer.fit(self.Xtrain, self.ytrain)
            if self.create_pkl is True:
                pkl_file = "models/{}_classifier_hu.pkl".format(self.trainer_name)
                trainer_pkl = open(pkl_file, "wb")
                pickle.dump(self.trainer, trainer_pkl)
                trainer_pkl.close()

    def test_predict(self):
        self.build_test_sets()
        p_test = self.trainer.predict(self.Xtest)
        p_train = self.trainer.predict(self.Xtrain)
        return p_train, p_test

class ActivityPredictor:
    def __init__(self, pkl_file=None, video_file=None, out_name=None):
        if pkl_file is not None:
            self.pkl_file = pkl_file
        else:
            self.pkl_file = "models/knn_classifier.pkl"
        self.labels = np.array([
            'boxing',
            'handclapping',
            'handwaving',
            'jogging',
            'running',
            'walking'
        ])
        self.video_file = video_file
        self.out_name = out_name
        trainer_pkl = open(self.pkl_file, "rb")
        self.trainer = pickle.load(trainer_pkl)
        self.total_frames = 0
        self.frame_count = 0
        self.fps = 30
        self.frame_size = (0, 0)
        self.predictions = []
        self.pred_tol = 10
        self.aq = ""

    def load_video_reader(self, filename):
        video = cv2.VideoCapture(filename)
        self.total_frames = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        self.frame_count = 0
        self.fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        self.frame_size = (int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
        return video

    def get_video(self, video):
        while video.isOpened():
            ret, frame = video.read()
            if ret:
                yield frame
            else:
                break
            self.frame_count += 1

        video.release()
        yield None

    def load_video_writer(self, fps, frame_size):
        self.video_name = "output_videos/classified_" + self.out_name + ".mp4"
        return cv2.VideoWriter(self.video_name, cv2.cv.CV_FOURCC(*'MP4V'), fps=fps, frameSize=frame_size, isColor=True)

    def get_mei_mhi(self, frames, aq, taus, tau_keys):
        prev_frame = aq.prepare_frame(frames[-1])

        mei_agg = np.zeros(prev_frame.shape, dtype=np.int32)
        mhi_image = np.zeros(prev_frame.shape, dtype=np.int32)

        for i, frame in enumerate(frames):
            if frame is None:
                return None

            morph_frame = aq.prepare_frame(frame)
            mei_frame = aq.binary_mei(morph_frame, prev_frame)
            mhi_image = aq.mhi(mei_frame, mhi_image)

            mei_agg += mei_frame

            prev_frame = morph_frame
            if i in taus:
                idx = taus.index(i)
                key = tau_keys[idx]
                if (np.sum(mei_agg) == 0.):
                    continue
                if (np.sum(mhi_image) == 0.):
                    continue
                if self.do_prediction(mhi_image, key) != -1:
                    mei_agg[mei_agg > 1] = 1
                    return mhi_image, mei_agg, key

            if (np.sum(mei_agg) == 0.):
                continue
            if (np.sum(mhi_image) == 0.):
                continue

        mei_agg[mei_agg > 1] = 1

        return mhi_image, mei_agg, None

    def get_best_prediction(self, prediction):
        self.predictions.append(prediction)

        if len(self.predictions) <= self.pred_tol + 1:
            return prediction
        else:
            if np.sum(np.array(self.predictions[-self.pred_tol:]) == prediction) > (self.pred_tol-2):
                return prediction
            else:
                return np.bincount(np.array(np.array(self.predictions)[-self.pred_tol:])).argmax()

    def do_prediction(self, mhi, tau_key):
        X_predict = []
        hus = self.aq.hu_moments(mhi)
        X_predict.append(hus)
        prediction = self.trainer.predict(X_predict)
        if self.labels[prediction[0]] == tau_key:
            return tau_key
        else:
            return -1

    def predict(self):
        self.aq = aq = ActivityQuantifier(filename=self.video_file)
        taus = aq.taus.values()
        tau_keys = aq.taus.keys()
        video = self.load_video_reader(filename=self.video_file)
        image_gen = self.get_video(video)
        video_writer = self.load_video_writer(self.fps, self.frame_size)
        origin = (0, self.frame_size[1]-5)
        fc_origin = (self.frame_size[0]-50, self.frame_size[1]-5)
        frames = []
        start = True
        frame_count = 0
        tt_times = []
        mom_times = []
        meimhi_times = []
        m = int(max(aq.taus.values())) + 1
        done = False
        label = "[ ]"
        curr_pred = len(self.labels)
        for _ in range(m):
            f = image_gen.next()
            frames.append(f)
        while len(frames) > 0:
            start1 = time.time()
            mhi, mei, poss_pred = self.get_mei_mhi(frames, aq, taus, tau_keys)
            end1 = time.time()
            getmeimhitime = end1 - start1
            meimhi_times.append(getmeimhitime)
            if (np.sum(mhi) != 0.):
                if poss_pred is not None:
                    label = "[ {} ]".format(poss_pred)
                    ptau = int(aq.taus[poss_pred])
                    for f in range(ptau):
                        curr_pred = np.argwhere(self.labels == poss_pred).flatten()[0]
                        self.predictions.append(curr_pred)
                        cv2.putText(frames[f], label, origin, cv2.FONT_HERSHEY_PLAIN, 2, 0)
                        cv2.putText(frames[f], str(frame_count), fc_origin, cv2.FONT_HERSHEY_PLAIN, 1, 0)
                        frame_count += 1
                        video_writer.write(frames[f])
                    frames = frames[ptau:]
                else:
                    X_predict = []
                    start2 = time.time()
                    hus = aq.hu_moments(mhi)
                    end2 = time.time()
                    moment_time = end2 - start2
                    mom_times.append(moment_time)
                    X_predict.append(hus)
                    prediction = self.trainer.predict(X_predict)
                    best_prediction = self.get_best_prediction(prediction[0])
                    if best_prediction == len(self.labels):
                        label = "[ ]"
                    else:
                        label = "[ " + self.labels[best_prediction] + " ]"
                    for f in frames:
                        cv2.putText(f, label, origin, cv2.FONT_HERSHEY_PLAIN, 2, 0)
                        cv2.putText(f, str(frame_count), fc_origin, cv2.FONT_HERSHEY_PLAIN, 1, 0)
                        frame_count += 1
                        self.predictions.append(best_prediction)
                        curr_pred = best_prediction
                        video_writer.write(f)
                    frames = []
            else:
                self.predictions.append(len(self.labels))
                curr_pred = len(self.labels)
                label = "[ ]"
                cv2.putText(frames[0], label, origin, cv2.FONT_HERSHEY_PLAIN, 2, 0)
                cv2.putText(frames[0], str(frame_count), fc_origin, cv2.FONT_HERSHEY_PLAIN, 1, 0)
                video_writer.write(frames[0])

                frames = frames[1:]
                frame_count += 1

            while len(frames) <= m:
                frame = image_gen.next()

                if frame is None:
                    for f in frames:
                        cv2.putText(f, label, origin, cv2.FONT_HERSHEY_PLAIN, 2, 0)
                        cv2.putText(f, str(frame_count), fc_origin, cv2.FONT_HERSHEY_PLAIN, 1, 0)
                        frame_count += 1
                        self.predictions.append(curr_pred)
                        video_writer.write(f)
                    done = True
                    break
                frames.append(frame)
                end3 = time.time()
                tt = end3 - start1
                tt_times.append(tt)
            if done:
                break
        video_writer.release()
        print "Moment time:", np.average(mom_times)
        print "Get Mei Mhi time:", np.average(meimhi_times)
        print "e2e times:", np.average(tt_times)
        return self.video_name, self.predictions

