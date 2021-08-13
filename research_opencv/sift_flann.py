import cv2
import time


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


# read original image
original_image = cv2.imread("bike.jpg")

# image to compare
image_to_compare = cv2.imread("bike/bike_0_2692.jpg")

print("Shape of original image:", original_image.shape)
print("Shape of compare image:", image_to_compare.shape)


sim = sift_sim(original_image, image_to_compare)

print("SIFT similarity", sim)

# result = cv2.drawMatches(original_image, kp_1, image_to_compare, kp_2, good_points, None)
# cv2.imshow("result", cv2.resize(result,None, fx=0.1, fy=0.1))

# cv2.imshow("original", cv2.resize(original_image,None, fx=0.2, fy=0.2))
# cv2.imshow("compare_image", cv2.resize(image_to_compare,None, fx=0.2, fy=0.2))



# cv2.waitKey(0)
# cv2.destroyAllWindows()