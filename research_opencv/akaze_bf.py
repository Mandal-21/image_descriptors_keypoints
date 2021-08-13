import cv2

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


original_img = cv2.imread('bike.jpg')
image_to_compare = cv2.imread('bike/bike_0_3855.jpg')

print("Shape of original image:", original_img.shape)
print("Shape of compare image:", image_to_compare.shape)


akaze_similarity = akaze_sim(original_img, image_to_compare) 

print("Similarity using AKAZE is: ", akaze_similarity)