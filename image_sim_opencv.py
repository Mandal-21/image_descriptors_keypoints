import cv2
import time
import tensorflow as tf
from skimage.metrics import structural_similarity


def sift_sim(img1, img2):

    '''
        Find the similarity between the images using SIFT (scale-invariant feature transform)
        and match through FLANN

        :parameter 
            img1: array of image 1
            img2: array of image 2

        :returns score of similarity
    '''


    # SIFT (scale-invariant feature transform)
    sift = cv2.SIFT_create()

    kp_1, desc_1 = sift.detectAndCompute(img1, None)
    kp_2, desc_2 = sift.detectAndCompute(img2, None)

    t0 = time.time()

    # Flann (Fast Library for Approximate Nearest Neighbors)
    index_params = dict(algorithm=0, trees=3)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc_1, desc_2, k=2)

    print(time.time() - t0)


    good_points = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good_points.append(m)


    number_keypoints = 0
    if len(kp_1) >= len(kp_2):
        number_keypoints = len(kp_1)
    else:
        number_keypoints = len(kp_2)

    print("len of keypoint",number_keypoints)
    print("len of good points", len(good_points))

    
    return len(good_points)/number_keypoints



def akaze_sim(img1, img2):

    '''
        Find the similarity between the images using AKAZE
        and match through Brute-Force

        :parameter 
            img1: array of image 1
            img2: array of image 2

        :returns score of similarity
    '''

    akaze = cv2.AKAZE_create()
    kpts1, desc1 = akaze.detectAndCompute(img1, None)
    kpts2, desc2 = akaze.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    
    similar_regions = [i for i in matches if i.distance < 50]
    if len(matches) == 0:
        return 0
    return len(similar_regions) / len(matches)


def tf_sim(img1, img2):

    '''
        Find the similarity between the images using Tensorflow sim method
        Note: image of different shapes are not applicable, you need to first make them of equal size

        :parameter 
            img1: array of image 1
            img2: array of image 2

        :returns score of similarity
    '''

    tf.shape(img1)  # `img1.jpg` has 3 channels; shape is `(255, 255, 3)`
    tf.shape(img2)  # `img2.jpg` has 3 channels; shape is `(255, 255, 3)`
    # Add an outer batch for each image.
    im1 = tf.expand_dims(img1, axis=0)
    im2 = tf.expand_dims(img2, axis=0)
    # Compute SSIM over tf.uint8 Tensors.
    ssim1 = tf.image.ssim(img1, img2, max_val=255, filter_size=11,
                            filter_sigma=1.5, k1=0.01, k2=0.03)

    # Compute SSIM over tf.float32 Tensors.
    im1 = tf.image.convert_image_dtype(im1, tf.float32)
    im2 = tf.image.convert_image_dtype(im2, tf.float32)
    ssim2 = tf.image.ssim(im1, im2, max_val=1.0, filter_size=11,
                            filter_sigma=1.5, k1=0.01, k2=0.03)
    # ssim1 and ssim2 both have type tf.float32 and are almost equal.
    return ssim1, ssim2


def orb_sim(img1, img2):

  '''
        Find the similarity between the images using ORB (Oriented FAST and Rotated BRIEF)
        and match through Brute-Force
        Note: Works well with images of different dimensions

        :parameter 
            img1: array of image 1
            img2: array of image 2

        :returns score of similarity
    '''

  orb = cv2.ORB_create()

  # detect keypoints and descriptors
  kp_a, desc_a = orb.detectAndCompute(img1, None)
  kp_b, desc_b = orb.detectAndCompute(img2, None)

  # define the bruteforce matcher object
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

  #perform matches. 
  matches = bf.match(desc_a, desc_b)
  #Look for similar regions with distance < 50. Goes from 0 to 100 so pick a number between.
  similar_regions = [i for i in matches if i.distance < 50]
  if len(matches) == 0:
    return 0
  return len(similar_regions) / len(matches)


def structural_sim(img1, img2):

    '''
        Find the similarity between the images using SKIMAGE  structural_similarity
        Note: Needs images to be same dimensions

        :parameter 
            img1: array of image 1
            img2: array of image 2

        :returns score of similarity
    '''

    sim, diff = structural_similarity(img1, img2, full=True, multichannel=True)
    return sim



if __name__ == "__main__":

    # read original image
    original_image = cv2.imread("bike.jpg")

    # image to compare
    image_to_compare = cv2.imread("bike/bike_0_9184.jpg")

    print("Shape of original image:", original_image.shape)
    print("Shape of compare image:", image_to_compare.shape)

    sim = tf_sim(original_image, image_to_compare)
    print(sim)