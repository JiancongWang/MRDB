# Second preprocess script. Merge them together after done. 
import collections
import numpy as np
import scipy.ndimage as ndi

#%% Try downsampling the image
def _nd_window(data, filter_function=np.hanning, inversed=False, epsilon=1e-20, rate=2.0):
    """
    Performs an in-place windowing on N-dimensional spatial-domain data.
    This is done to mitigate boundary effects in the FFT.
    Parameters
    ----------
    data : ndarray
           Input data to be windowed, modified in place.
    filter_function : 1D window generation function
           Function should accept one argument: the window length.
           Example: scipy.signal.hamming
    Credits for:

    dsp.stackexchange.com/questions/19519/extending-1d-window-functions-to-3d-or-higher
    Modified by Yuhua Bill CHEN
    """
    for axis, axis_size in enumerate(data.shape):
        # set up shape for numpy broadcasting
        filter_shape = [1, ] * data.ndim
        filter_shape[axis] = axis_size
        window = filter_function(axis_size * rate) + epsilon  # Undersampled by ratio
        window = np.power(window, (1.0 / data.ndim))
        length = axis_size
        startx = int(axis_size * rate // 2 - length // 2)
        window = window[startx:startx + length]
        if inversed:
            window = 1 / window
        window = window.reshape(filter_shape)

        data *= window
    return data


def downsample_image(full_img,
                     window_function=np.hanning,
                     size_ratio=(2, 1, 2),
                     unfilter_window_size=1.25,
                     filter_window_size=1.5):
    """
    Down sampling image in kspace
    :param full_img:            Full resolution image
    :param window_function:     Window function for filtering
    :param size_ratio:          Down-sampling ratio, default is (2, 1, 2), where along the 2nd dimension resolution
                                is preserved.
                                (2, 2, 2) will have a resolution reduction of 2 along all dimensions
    :param unfilter_window_size: Window size of invert filtering # To mimic MRI scanner
    :param filter_window_size:  Window size of filtering         # To mimic MRI scanner
    :return: downsampled_image
    """

    # As for July 2018, turns out to be the scanner is just using interpolation instead of zero-filling k-space
    # So the program change to use interpolation method, away from zero-filling

    # Stacked 2d slice in axes(0,1)
    

    if not isinstance(size_ratio, collections.Sequence):
        size_ratio = (size_ratio, 1, size_ratio)

    dtype = full_img.dtype
    full_kspace = np.fft.fftshift(np.fft.fftn(full_img))

    # Boosted k-space
    bt_kspace = _nd_window(np.array(full_kspace),
                                filter_function=window_function,
                                inversed=True, rate=unfilter_window_size)

    # Crop k-space
    dim0_cropped_slice_num = int(full_kspace.shape[0] / size_ratio[0])
    dim0_padding = int((full_kspace.shape[0] - dim0_cropped_slice_num) / 2)
    dim1_cropped_slice_num = int(full_kspace.shape[1] / size_ratio[1])
    dim1_padding = int((full_kspace.shape[1] - dim1_cropped_slice_num) / 2)
    dim2_cropped_slice_num = int(full_kspace.shape[2] / size_ratio[2])
    dim2_padding = int((full_kspace.shape[2] - dim2_cropped_slice_num) / 2)

    cropped_kspace = bt_kspace[
                     dim0_padding:dim0_padding + dim0_cropped_slice_num,
                     dim1_padding:dim1_padding + dim1_cropped_slice_num,
                     dim2_padding:dim2_padding + dim2_cropped_slice_num
                     ]

    # Smoothed k-space
    sm_kspace = _nd_window(np.array(cropped_kspace),
                                filter_function=window_function,
                                inversed=False, rate=filter_window_size) / np.product(
        np.divide(full_kspace.shape, cropped_kspace.shape))

    us_img = np.fft.ifftn(np.fft.ifftshift(sm_kspace))
    if np.iscomplexobj(full_img):
        us_real = np.real(us_img)
        us_imag = np.imag(us_img)

        us_real = ndi.zoom(
            us_real,
            (float(full_kspace.shape[0] / cropped_kspace.shape[0]),
             float(full_kspace.shape[1] / cropped_kspace.shape[1]),
             float(full_kspace.shape[2] / cropped_kspace.shape[2])),
            mode='mirror',
            order=1
        )
        us_imag = ndi.zoom(
            us_imag,
            (float(full_kspace.shape[0] / cropped_kspace.shape[0]),
             float(full_kspace.shape[1] / cropped_kspace.shape[1]),
             float(full_kspace.shape[2] / cropped_kspace.shape[2])),
            mode='mirror',
            order=1
        )

        us_img = (us_real + us_imag * 1j)

    else:
        us_img = np.abs(us_img)

        us_img = ndi.zoom(
            us_img,
            (float(full_kspace.shape[0] / cropped_kspace.shape[0]),
             float(full_kspace.shape[1] / cropped_kspace.shape[1]),
             float(full_kspace.shape[2] / cropped_kspace.shape[2])),
            mode='mirror',
            order=1
        )

        us_img = np.maximum(0.0, us_img)  # Remove possible negative values from interpolation

    us_img = us_img.astype(dtype=dtype)
    return us_img