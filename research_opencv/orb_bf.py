import cv2

#Works well with images of different dimensions
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




original_img = cv2.imread('bike.jpg')
image_to_compare = cv2.imread('bike/bike_0_3855.jpg')

print("Shape of original image:", original_img.shape)
print("Shape of compare image:", image_to_compare.shape)


orb_similarity = orb_sim(original_img, image_to_compare) 

print("Similarity using ORB is: ", orb_similarity)
