import os
import argparse
import warnings

import rasterio
import numpy as np


warnings.filterwarnings("ignore",
                        category=rasterio.errors.NotGeoreferencedWarning)


def fourier_zoom(image, z=2):
    h, w = image.shape

    left = np.ceil((z - 1) * w / 2).astype(int)
    right = np.floor((z - 1) * w / 2).astype(int)
    top = np.ceil((z - 1) * h / 2).astype(int)
    bottom = np.floor((z - 1) * h / 2).astype(int)

    ft = np.fft.fftshift(np.fft.fft2(image))

    ft = np.pad(ft, [(top, bottom), (left, right)], 'constant', constant_values=0+0j)

    out = np.fft.ifft2(np.fft.ifftshift(ft))

    if np.isrealobj(image):
        out = np.real(out)

    return out * z * z


def _test_fourier_zoom(w, h, z=2):
    image = (2 * np.random.random((h, w)) - 1) * 1e3

    zoomed_image = fourier_zoom(image, z)

    
    np.testing.assert_allclose(zoomed_image[::z, ::z], image)


def test_fourier_zoom():
    """
    """
    for w in range(1, 8):
        for h in range(1, 8):
            for z in range(1, 8):
                _test_fourier_zoom(w, h, z)


def rasterio_write(path, array, profile={}, tags={}):
    
    extension = os.path.splitext(path)[1].lower()
    if extension in ['.tif', '.tiff']:
        driver = 'GTiff'
    elif extension in ['.png']:
        driver = 'png'
    else:
        raise NotImplementedError('format {} not supported'.format(extension))

    # read image size and number of bands
    array = np.atleast_3d(array)
    height, width, nbands = array.shape

    # define image metadata dict
    profile.update(driver=driver, count=nbands, width=width, height=height,
                   dtype=array.dtype)

    # write to file
    with rasterio.Env():
        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(np.transpose(array, (2, 0, 1)))
            dst.update_tags(**tags)


def main(input_image_path, output_image_path, zoom_factor):
    """
    """
    with rasterio.open(input_image_path, 'r') as f:
        img = f.read().squeeze()

    rasterio_write(output_image_path, fourier_zoom(img, zoom_factor))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Fourier zoom'))
    parser.add_argument('input_image', help=('path to the input image file'))
    parser.add_argument('output_image', help=('path to the output image file'))
    parser.add_argument('zoom', type=int, help=('zoom factor'))
    args = parser.parse_args()
    main(args.input_image, args.output_image, args.zoom)
