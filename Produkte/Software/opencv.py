import cv2
import numpy as np

    

def main():
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("Channels")
    cv2.namedWindow("Channels_HSV")
    cv2.namedWindow("Channels_YCrCb")

    cv2.namedWindow("Camera")
    #skin color training

    counter = 0
    train_frames = []
    # for i in range(10):
    #     counter += 1
    #     ret, frame = cam.read()
    #     train_frames.append(frame)

    #     cv2.imshow("Camera", frame)

    #     if not ret:
    #         break
        
    #     k = cv2.waitKey(1)

    #     if k%256 == 27:
    #         # esc key
    #         print("Escape was pressed...")
    #         break

    # median = np.median(train_frames, axis=0).astype(dtype=np.uint8)
    # median_hsv = cv2.cvtColor(median, cv2.COLOR_BGR2HSV)

    while True:
        ret, frame = cam.read()
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

        r,g,b = cv2.split(frame)
        h,s,v = cv2.split(frame_hsv)       
        yrb,cr,cb = cv2.split(frame_ycrcb)

        mask = np.array([[255]*frame.shape[1]]*frame.shape[0], dtype=np.uint8)
        for x in range(frame.shape[1]):
            for y in range(frame.shape[0]):
                if (r[y,x] > 95 and g[y,x] > 40 and b[y,x] > 20 and r[y,x] > g[y,x] and r[y,x] > b[y,x]
                    and abs(r[y,x] - g[y,x]) > 15 
                    and cr[y,x] > 135 and cb[y,x] > 85 and yrb[y,x] > 80 and cr[y,x] <= (1.5862*cb[y,x])+20 
                    and cr[y,x] >=(0.3448*cb[y,x])+76.2069 and cr[y,x] >= (-4.5652* cb[y,x])+234.5652 
                    and cr[y,x] <= (-1.15*cb[y,x])+301.75 and cr[y,x] <= (-2.2857*cb[y,x])+432.85):
                        mask[y,x] = 0


        frame2 = cv2.bitwise_and(frame, frame, mask=mask)
        full = np.array([[255]*h.shape[1]]*h.shape[0], dtype=np.uint8)
        frame3 = cv2.merge([h, full, full])
        frame3 = cv2.cvtColor(frame3, cv2.COLOR_HSV2BGR)

        frame4 = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        display_channels("Channels", frame)
        display_channels("Channels_HSV", frame_hsv)
        display_channels("Channels_YCrCb", frame4)
        cv2.imshow("Camera", frame2)
        if not ret:
            break
        
        k = cv2.waitKey(1)

        if k%256 == 27:
            # esc key
            print("Escape was pressed...")
            break

def display_channels(window_name, img):
    channels = cv2.split(img)
    result = np.concatenate(channels, axis=1)
    cv2.imshow(window_name, result)


def hog(frame):
    winSize = (20,20)
    blockSize = (10,10)
    blockStride = (5,5)
    cellSize = (10,10)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    useSignedGradients = True

    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, useSignedGradients)
    descriptor = hog.compute(frame)

if __name__ == "__main__":
    main()