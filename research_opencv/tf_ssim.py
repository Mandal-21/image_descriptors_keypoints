import tensorflow as tf


# image of different shapes are not applicable, you need to first make them of equal size
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

original_img = tf.image.decode_image(tf.io.read_file('bike.jpg'))
image_to_compare = tf.image.decode_image(tf.io.read_file('bike/bike_0_2692.jpg'))

ssim1, ssim2 = tf_sim(original_img, image_to_compare)
print(ssim1, ssim2)