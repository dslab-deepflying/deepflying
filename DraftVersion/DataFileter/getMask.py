import cv2
import numpy as np
threshold = [90, 90, 90]

def fill_color_demo(image):
    copyImg = image.copy()
    h, w = image.shape[:2]
    mask = np.zeros([h+2, w+2],np.uint8)

    w1, h1, w2, h2 = int(0.1*w), int(0.1*h),int(0.9 * w), int(0.9 * h)

    # seed upper left
    seed_ul = (w1, h1)
    # seed bottom right
    seed_br = (w2, h2)

    # mean value of point
    mean = [0, 0, 0]
    mean[0] = image.item(h1, w1, 0) + image.item(h2, w2, 0)
    mean[1] = image.item(h1, w1, 1) + image.item(h2, w2, 1)
    mean[2] = image.item(h1, w1, 2) + image.item(h2, w2, 2)

    mean[0] /= 2
    mean[1] /= 2
    mean[2] /= 2

    loDiff = [a + b for a, b in zip(mean, threshold)]
    upDiff = [a - b for a, b in zip(mean, threshold)]

    flags = 4 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE

    print(loDiff,upDiff)



    cv2.floodFill(copyImg, mask, seed_ul, (0, 0, 0), threshold, threshold, cv2.FLOODFILL_FIXED_RANGE)
    cv2.floodFill(copyImg, mask, seed_br, (0, 0, 0), threshold, threshold, cv2.FLOODFILL_FIXED_RANGE)

    cv2.imshow("fill_color_demo", copyImg)

def main():
    src = cv2.imread('1.jpg')
    cv2.namedWindow('input_image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('input_image', src)
    fill_color_demo(src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

