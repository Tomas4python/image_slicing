import numpy as np
from src.main import concatenate_arrays, apply_color_scheme, Settings


def test_concatenate_arrays():

    array1 = np.ones((6, 5))
    array2 = np.ones((5, 5))
    array3 = np.ones((5, 6))
    array4 = np.ones((5, 5))
    concatenated_array = concatenate_arrays((array1, array2, array3, array4))
    assert concatenated_array.shape == (10, 10)


def test_apply_color_scheme():

    img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
    Settings.colour_scheme = 'red'
    transformed_img = apply_color_scheme(img)
    assert transformed_img.shape == img.shape
    assert np.all(transformed_img[:, :, 1] == 0), "Green channel contains non-zero values."
    assert np.all(transformed_img[:, :, 2] == 0), "Blue channel contains non-zero values."
    assert np.array_equal(transformed_img[:, :, 0], img[:, :, 0]), "Red channel values have been altered."
