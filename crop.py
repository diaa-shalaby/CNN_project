import numpy as np


def crop_image(image_array: np.ndarray, border_x: tuple, border_y: tuple) -> tuple:
    """
    Creates two input arrays and one target array from one input image. For this, this function
    removes (=set values to zero) the specified borders in the image. These removed values will
    later become the target, i.e. our network has to learn to recreate the values that we set to
    zero.

    @param image_array: A numpy array of shape (X, Y) and arbitrary datatype, which contains
    the image data.
    @param border_x: A tuple containing 2 int values. These two values specify the border thickness at
    the start and end of the x axis of the image_array, respectively. Please see the figure below for
    more information
    @param border_y: A tuple containing 2 int values. These two values specify the border thickness
    at the start and end of the y axis of the image_array, respectively. Please see the
    figure below for more information.

    @return input_array: should be a 2D numpy array of same shape and datatype as image_array. It
    should have the same pixel values as image_array, with the exception that the to-be-removed
    pixel values in the specified borders are set to 0. You may edit the original image_array
    in-place or create a copy.
    @return known_array: 2D numpy array of same shape and datatype as image_array, where
    pixels in the specified borders should have value 0 and other, known, pixels have value 1.
    @return target_array: 1D numpy array of the same datatype as image_array. It should
    hold the pixel values of the specified borders (the pixels that were set to 0 in input_array).
    The order of the pixels in target_array should be the same as if one would use known_array
    as boolean mask on image_array (like image[boolean_mask]). The length of target_array
    should therefore be the number of pixels in the specified borders.
    """
    # Raise an exception if image_array is not a 2D numpy array
    if not isinstance(image_array, np.ndarray) or image_array.ndim != 2:
        raise NotImplementedError

    # Raise ValueError if the values in borders are not convertible to int.
    try:
        border_x = (int(border_x[0]), int(border_x[1]))
        border_y = (int(border_y[0]), int(border_y[1]))
    except ValueError:
        raise ValueError("The values in borders are not convertible to int.") from None

    # Raise a ValueError if the values are smaller than 1.
    if border_x[0] < 1 or border_x[1] < 1:
        raise ValueError("The values in border_x should be greater than 1")
    if border_y[0] < 1 or border_y[1] < 1:
        raise ValueError("The values in border_y should be greater than 1")

    # Raise a ValueError if the shape of the remaining known image pixels would be smaller than (16, 16).
    diff_x = border_x[0] + border_x[1]
    diff_y = border_y[0] + border_y[1]
    if image_array.shape[0] - diff_x < 16 or image_array.shape[1] - diff_y < 16:
        raise ValueError("The remaining known image pixels after removing the borders would be too small")


    # Create input array from a copy of image array and modify this copy
    input_array = image_array.copy()

    input_array[:, 0:border_y[0]] = 0
    input_array[:, - border_y[1]:] = 0

    input_array[0:border_x[0], :] = 0
    input_array[- border_x[1]:, :] = 0

    # Initiate know_array with all zeros but with same shape and dtype of image_array.
    known_array = np.zeros_like(image_array)

    known_array[border_x[0]: -border_x[1], border_y[0]: -border_y[1]] = 1

    # target_array: image_array[boolean_mask=known_array] to get the the removed pixels from the image in a 1D array.
    target_array = image_array[known_array == 0]

    return input_array, known_array, target_array


