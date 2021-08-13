import cv2
from skimage.metrics import structural_similarity

#Needs images to be same dimensions
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



original_img = cv2.imread('bike.jpg')
image_to_compare = cv2.imread('bike/bike_0_3855.jpg')

print("Shape of original image:", original_img.shape)
print("Shape of compare image:", image_to_compare.shape)

ssim = structural_sim(original_img, image_to_compare)
print("Similarity using SSIM is: ", ssim)