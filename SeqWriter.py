# (c) 2020-04-03 Martin Matousek
# Last change: $Date$
#              $Revision$

import cv2
import numpy as np
import os.path


class SeqWriter:

    def __init__(self, filename):
        self.video = None
        self.decimate = 1
        self.filename = filename
        self.frameno = 0
        _, ext = os.path.splitext(filename)

        if ext == '.avi':
            self.video = -1
            return

        raise Exception('Unhandled file type:' + filename)

    def Close(self):
        if self.video is not None:
            self.video.release()
            self.video = None

    def Write(self, frame):
        if self.decimate > 1:
            d = int(np.ceil(self.decimate))

            nframe = 0.0
            for i in range(0, d):
                for j in range(0, d):
                    nframe += frame[i:-d + i:d, j:-d + j:d, :].astype(float)

            nframe /= float(d * d)
            frame = nframe.astype(np.uint8)

        if self.video == -1:
            mjpg = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            self.video = cv2.VideoWriter(self.filename, mjpg, 25,
                                         (frame.shape[1], frame.shape[0]))

        self.frameno += 1

        if self.video is not None:
            frame_bgr = np.empty_like(frame)
            frame_bgr[:, :, 0] = frame[:, :, 2]
            frame_bgr[:, :, 1] = frame[:, :, 1]
            frame_bgr[:, :, 2] = frame[:, :, 0]

            self.video.write(frame_bgr)