import os
import urllib
import zipfile

ACTIONS = [
    "boxing",
    "handclapping",
    "handwaving",
    "jogging",
    "running",
    "walking"
]

def download_dataset():
    print "Getting zip videos..."
    urllib.urlretrieve("http://www.nada.kth.se/cvap/actions/walking.zip", "zip_videos/walking.zip")
    print "Done downloading 1 of 6"
    urllib.urlretrieve("http://www.nada.kth.se/cvap/actions/boxing.zip", "zip_videos/boxing.zip")
    print "Done downloading 2 of 6"
    urllib.urlretrieve("http://www.nada.kth.se/cvap/actions/handclapping.zip", "zip_videos/handclapping.zip")
    print "Done downloading 3 of 6"
    urllib.urlretrieve("http://www.nada.kth.se/cvap/actions/handwaving.zip", "zip_videos/handwaving.zip")
    print "Done downloading 4 of 6"
    urllib.urlretrieve("http://www.nada.kth.se/cvap/actions/jogging.zip", "zip_videos/jogging.zip")
    print "Done downloading 5 of 6"
    urllib.urlretrieve("http://www.nada.kth.se/cvap/actions/running.zip", "zip_videos/running.zip")
    print "Done downloading, now unzipping..."
    for a in ACTIONS:
        zip_ref = zipfile.ZipFile("zip_videos/" + a + ".zip", 'r')
        zip_ref.extractall("input_videos/" + a + "/")
        zip_ref.close()

def download_experiment():
    print "Getting experiment videos..."
    urllib.urlretrieve("https://www.rangeandroam.com/videos/Test1.mp4", "experiment_videos/Test1.mp4")
    urllib.urlretrieve("https://www.rangeandroam.com/videos/Test2.mp4", "experiment_videos/Test2.mp4")
    urllib.urlretrieve("https://www.rangeandroam.com/videos/Test3.mp4", "experiment_videos/Test3.mp4")
    urllib.urlretrieve("https://www.rangeandroam.com/videos/Test4.mp4", "experiment_videos/Test4.mp4")
    urllib.urlretrieve("https://www.rangeandroam.com/videos/Test5.mp4", "experiment_videos/Test5.mp4")
    print "Done getting experiment videos"

def download_pkl():
    print "Getting classifier pkl..."
    urllib.urlretrieve("https://www.rangeandroam.com/videos/knn_classifier.pkl", "models/knn_classifier.pkl")
    print "Done getting classifier pkl"

def run():
    print "Running setup..."
    dirs = ["output", "output_videos", "models", "input_videos", "zip_videos", "experiment_videos"]
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
    download_dataset()
    download_experiment()
    download_pkl()
    os.remove("input_videos/boxing/person01_boxing_d4_uncomp.avi")
    print "Done with setup!"
    print "======================"
    print "To run the experiment"
    print "$ python experiment.py"
    print "======================"

if __name__ == "__main__":
    run()