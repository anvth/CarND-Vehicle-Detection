from utils import Processing
import numpy as np
import cv2
import utils as aux
from moviepy.editor import ImageSequenceClip, VideoFileClip
import os.path
from tqdm import tqdm
from scanner import VehicleScanner


class Detector:
    def __init__(self, imgMarginWidth=320, historyDepth=5, margin=100, windowSplit=2, winCount=9,
                 searchPortion=1., veHiDepth=30, pointSize=64,
                 groupThrd=10, groupDiff=.1, confidenceThrd=.7):
        self.imgProcessor = Processing()
        self.imgMarginWidth = imgMarginWidth
        
        self.scanner = VehicleScanner(pointSize=pointSize,
                                      veHiDepth=veHiDepth, groupThrd=groupThrd, groupDiff=groupDiff,
                                      confidenceThrd=confidenceThrd)

    def addPip(self, pipImage, dstImage, pipAlpha=0.5, pipResizeRatio=0.3, origin=(20, 20)):
        """
        Adding small Picture-in-picture binary bird-eye projection with search areas and found lines embedded
        :param pipImage: original binary bird-eye projection with search areas and found lines embedded
        :param dstImage: destination color image (assumed undistorted)
        :param pipAlpha: pip alpha
        :param pipResizeRatio: pip scale
        :param origin: coordinates of upper-left corner of small picture
        :return: color image with P-i-P embedded
        """
        smallPip = self.imgProcessor.resize(src=pipImage, ratio=pipResizeRatio)

        pipHeight = smallPip.shape[0]
        pipWidth = smallPip.shape[1]

        backGround = dstImage[origin[1]:origin[1] + pipHeight, origin[0]:origin[0] + pipWidth]

        blend = np.round(backGround * (1 - pipAlpha), 0) + np.round(smallPip * pipAlpha, 0)

        blend = np.minimum(blend, 255)

        dstImage[origin[1]:origin[1] + pipHeight, origin[0]:origin[0] + pipWidth] = blend

        # return dstImage

    def embedDetections(self, src, pipParams=None):
        """
        Main 'pipeline' for adding Lane polygon AND detected vehicles to the original image
        :param src: original image
        :param pipParams: alpha and scale ratios
        :return: undistorted color image with the Lane embedded.
        """
        img = self.imgProcessor.undistort(src=src)

        vBoxes, heatMap = self.scanner.relevantBoxes(src=img)

        
        aux.drawBoxes(img=img, bBoxes=vBoxes)

        # Upper left corner where starting to add pip and telemetry
        origin = (20, 20)

        # Adding PIP
        if pipParams is not None:
            alpha = pipParams['alpha']
            ratio = pipParams['scaleRatio']

            # To keep for subsequent telemetry stamps
            heatWidth = int(heatMap.shape[1] * ratio)

            # Vehicle Detection Picture-in-Picture
            self.addPip(pipImage=heatMap, dstImage=img,
                        pipAlpha=alpha, pipResizeRatio=ratio,
                        origin=(img.shape[1] - heatWidth - 20, 20))

        return img


def main():
    """
    Runs when invoking directly from command line
    :return: 
    """
    resultFrames = []

    clipFileName = input('Enter video file name: ')

    if not os.path.isfile(clipFileName):
        print('No such file. Exiting.')
        return

    clip = VideoFileClip(clipFileName)

    # depth = aux.promptForInt(message='Enter history depth in frames: ')
    # detectionPointSize = aux.promptForInt(message='Enter Search Margin: ')
    # fillerWidth = aux.promptForInt(message='Enter filler width: ')
    # windowSplit = aux.promptForInt(message='Enter Window Split: ')
    # winCount = aux.promptForInt(message='Enter Window Count for Box Search: ')
    # searchPortion = aux.promptForFloat(message='Enter the Search portion (0.0 - 1.0): ')
    # pipAlpha = aux.promptForFloat(message='Enter Picture-in-picture alpha: (0.0 - 1.0): ')
    # pipScaleRatio = aux.promptForFloat(message='Enter Picture-in-picture scale (0.0 - 1.0): ')

    depth = 5
    margin = 100
    fillerWidth = 320
    windowSplit = 2
    winCount = 18
    searchPortion = 1.

    pipAlpha = .7
    pipScaleRatio = .35

    pipParams = {'alpha': pipAlpha, 'scaleRatio': pipScaleRatio}

    print('Total frames: {}'.format(clip.duration * clip.fps))

    ld = Detector(imgMarginWidth=fillerWidth, historyDepth=depth,
                  margin=margin, windowSplit=windowSplit, winCount=winCount,
                  searchPortion=searchPortion, veHiDepth=45,
                  pointSize=64, groupThrd=10, groupDiff=.1, confidenceThrd=.5)

    for frame in tqdm(clip.iter_frames()):
        print(type(frame))
        dst = ld.embedDetections(src=frame, pipParams=pipParams)
        resultFrames.append(dst)

    resultClip = ImageSequenceClip(resultFrames, fps=25, with_mask=False)
    resultFileName = clipFileName.split('.')[0]
    resultFileName = '{}_out_{}.mp4'.format(resultFileName, aux.timeStamp())
    resultClip.write_videofile(resultFileName, progress_bar=True)


if __name__ == '__main__':
    main()