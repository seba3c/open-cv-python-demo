import os
import cv2

SAMPLES_DIR = "samples"
OUTPUT_DIR = "output"

SAMPLE = "fairchild_republic_A-10_thunderbolt _II.avi"

input_file = os.path.join(SAMPLES_DIR, SAMPLE)
output_file = os.path.join(OUTPUT_DIR, "SIFT_%s" % SAMPLE)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(output_file, fourcc, 20.0, (640, 360))

cap = cv2.VideoCapture(input_file)

sift = cv2.xfeatures2d.SIFT_create()

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret:
        # Our operations on the frame come here
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kp = sift.detect(frame, None)

        frame = cv2.drawKeypoints(frame, kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                                  outImage=frame)

        out.write(frame)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
