import os
import cv2

SAMPLES_DIR = "samples"
OUTPUT_DIR = "output"

SAMPLE_1 = "super-etendar_1.jpg"
SAMPLE_2 = "super-etendar_2.jpg"
SAMPLE_3 = "pisa_tower.jpg"

samples = [SAMPLE_1, SAMPLE_2, SAMPLE_3]

surf = cv2.xfeatures2d.SURF_create()

for s in samples:

    file = os.path.join(SAMPLES_DIR, s)
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kp = surf.detect(img, None)

    img = cv2.drawKeypoints(img, kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                            outImage=img)

    output_file = os.path.join(OUTPUT_DIR, "SURF_%s" % s)
    cv2.imwrite(output_file, img)
