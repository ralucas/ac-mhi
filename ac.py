import sys

import ps7

try:
    video_file = sys.argv[1]
    output_name = sys.argv[2]
except:
    print "Video file is required: $ python ac.py [video_file] [output_name]"
    sys.exit(-1)


ap = ps7.ActivityPredictor(video_file=video_file, out_name=output_name)
new_video_file = ap.predict()
print "======================================"
print "Video created: " + str(new_video_file)
print "======================================"
