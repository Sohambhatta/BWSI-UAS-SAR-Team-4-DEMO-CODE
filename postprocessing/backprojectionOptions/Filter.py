import numpy as np
from scipy import ndimage
import cv2 as cv
import math


def enhanced_lee(image: np.ndarray, size: int):
    """
    Creating an image filter for sar images that is meant to remove speckle noise
    by taking the mean of the pixels around a certain center pixel and then applying a linear weight
    to its value to smooth out the image. However this is the enhanced version that has direction awareness, so it can detect
    edges through gradients and makes sure to retain edges by smoothing around then rather than smoothing over them.

    size: size of each kernel (bigger means more blurring)
    
    """
    # Getting the image dimensions and creating the processing kernels.
    height, width = image.shape
    ones = np.ones((size, size))
    kernel = ones / ones.sum()

    # Getting pixel means across the kernels
    local_mean = ndimage.convolve(image, kernel, mode="reflect")
    local_mean_sq = ndimage.convolve(image**2, kernel, mode="reflect")

    # Computing the variance for how different the pixels values are
    local_variance = local_mean_sq - local_mean**2
    noise_var = np.mean(local_variance)

    # Using a sobel filter to get the gradients for the image to be used for edge detection
    grad_x = ndimage.sobel(image, axis=1)
    grad_y = ndimage.sobel(image, axis=0)

    # Determing the direction/orientation of the edges
    Across_edge_direction = np.arctan2(grad_y, grad_x)
    Along_edge_direction = Across_edge_direction + math.pi / 2

    # Creating a new array for the filtered image
    filtered = np.zeros_like(image)

    # Precompute Lee weights (2D array)
    weight = (local_variance - noise_var) / (local_variance + 1e-12)
    weight = np.clip(weight, 0, 1)

    half_length = size // 2

    # Loop over every pixel (y=row, x=col)
    for y in range(height):
        for x in range(width):
            theta = Along_edge_direction[y, x]

            # 1) Gather neighbors along the edge direction
            vals = []
            for i in range(-half_length, half_length + 1):
                dx = i * math.cos(theta)
                dy = i * math.sin(theta)

                xi = int(round(x + dx))
                yi = int(round(y + dy))

                # clamp into bounds
                xi = max(0, min(width - 1, xi))
                yi = max(0, min(height - 1, yi))
                xi = max(0, min(width - 1, xi))
                yi = max(0, min(height - 1, yi))

                vals.append(image[yi, xi])

            # 2) Compute directional mean for this one pixel

            directional_mean = sum(vals) / len(vals)

            # 3) Fetch its Lee weight
            w = weight[y, x]

            # 4) Blend and write to the output
            filtered[y, x] = directional_mean + w * (image[y, x] - directional_mean)

    # After loops, return the full image
    return filtered


def enhanced_frost(image: np.ndarray, size: int, alpha: float):
    """
    Works the same way a lee filter algorithm does but instead of using a simple linear weight we use an
    exponential weighting format but the exponentially decaying weight does not have

    size: size of the kernel (bigger size means more smoothing)

    alpha: variable in the exponential weight decay function (bigger alpha means more decay over distance)

    """

    # Get Dimensions and create kernel
    height, width = image.shape
    half_length = size // 2
    ones = np.ones((size, size))
    kernel = ones / ones.sum()

    # Using a sobel filter to get the gradients for the image to be used for edge detection
    grad_x = ndimage.sobel(image, axis=1)
    grad_y = ndimage.sobel(image, axis=0)
    edge_direction = np.arctan2(grad_y, grad_x) + math.pi / 2

    # calculates different statistics of the image
    local_mean = ndimage.convolve(image, kernel)
    local_mean_sq = ndimage.convolve(image**2, kernel)
    local_variance = local_mean_sq - local_mean**2 + 1e-12
    noise_var = np.mean(local_variance)

    # Creates Image Array
    filtered = np.zeros_like(image)

    # Loops through pixels
    for y in range(height):
        for x in range(width):
            theta = edge_direction[y, x]
            vals = []
            weights = []

            for i in range(-half_length, half_length + 1):
                dx = i * math.cos(theta)
                dy = i * math.sin(theta)

                xi = int(round(x + dx))
                yi = int(round(y + dy))

                # clamp into bounds
                xi = max(0, min(width - 1, xi))
                yi = max(0, min(height - 1, yi))

                # gets Frost weights and adds to arrays
                vals.append(image[yi, xi])
                k = alpha * (noise_var / (local_variance[y, x] + 1e-12))
                weights.append(np.exp(-k * abs(i)))

            # Converts arrays to np.arrays
            vals = np.array(vals)
            weights = np.array(weights)

            filtered[y, x] = np.sum(weights * vals) / np.sum(weights)

    return filtered


# ============Filters to be used after despeckling image using the frost or lee=========================================


def gamma_MAP(image: np.ndarray, size: int, num_looks: int, upper_threshold=2.0):
    """
    Gamma Maximum A Posteriori (MAP) filter to reduce speckle noise and enhance edges
    This filter uses Bayesian statistics and the gamma distribution to adaptively denoise
    each pixel
    
    size: size of the kernel (bigger usually means more smoothing)

    num_looks: number of times the platform flies over the target

    upper_threshold: threshold after which an object is considered a point and not noise

    """

    # Getting the image stats
    kernel = np.ones((size, size)) / (size * size)

    # Getting different statistics needed for filter
    local_mean = ndimage.convolve(image, kernel, mode="reflect")
    local_mean_sq = ndimage.convolve(image**2, kernel, mode="reflect")
    local_variance = local_mean_sq - local_mean**2 + 1e-12

    output_image = np.empty_like(image)

    alpha = local_mean**2 / (local_variance - local_mean**2 / num_looks + 1e-12)
    coef_of_var = np.sqrt(local_variance) / (local_mean)
    speckle_idx = 1 / np.sqrt(num_looks)  # Only one since we only take one look

    lower_thresh = speckle_idx + 1e-3
    upper_thresh = upper_threshold  # will be changed accordingly

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            Y0 = image[i, j]
            Cv = coef_of_var[i, j]
            u = local_mean[i, j]
            alph = alpha[i, j]

            if Cv >= upper_thresh:  # Point-Target if true
                X = Y0
            elif Cv <= lower_thresh:  # Homogenous if true
                X = (num_looks * Y0 - (num_looks - 1) * u) / num_looks
            else:  # Textured
                X = ((alph + num_looks) * Y0 - (num_looks - 1) * u) / (
                    alph + num_looks - 1
                )

            output_image[i, j] = X

    return output_image


def Unsharp_Masking(image: np.ndarray, size: int, sigma=1.3, alpha=1.0):
    """
    Unsharp masking is another edge enhancement filtering algorithm. Essentially it works by creating gaussian based kernels
    for each part of the image. It convolves each pixel in the kernel with the gaussian kernel values and then isolates the image to
    retain only higher frequencies. We then take that isolated image and multiply it back to the original image to enhance the higher frequencies in the image
    typically the or around the edges.

    - Sigma = blur radius. Good starting point is between 1.0 - 2.0, going higher will produce more pronounced halos

    - Alpha = scalar_weight. Good starting point is between 0.5 - 2.0 and will give increasingly higher edge boost.

        BE CAREFUL OF HALOS WITH HIGH ALPHA VALUE
    """

    # Creating kernels for applying filters through image
    half_kernel = size // 2
    x = np.arange(-half_kernel, half_kernel + 1)
    y = np.arange(-half_kernel, half_kernel + 1)
    X, Y = np.meshgrid(x, y)

    # Turning the kernels to gaussian kernels and normalizing 
    gaussian_kernel = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    norm_kernel = gaussian_kernel / np.sum(gaussian_kernel)

    # filtering original image with the gaussian kernels 
    filtered = cv.filter2D(
        image, ddepth=-1, kernel=norm_kernel, borderType=cv.BORDER_REFLECT
    )

    # Isolating the image's higher frequencies 
    mask = image - filtered

    # Adding the higher intensities back into the image to boost the high intensity peaks in original image
    out_image = image + alpha * mask
    
    # Returning image
    return out_image

def Anisotropic_Diffusion(
    image, num_iterations: int = 10, kappa: float = 30.0, gamma: float = 0.1
):
    """
    Applies Perona-Malik anisotropic diffusion to reduce noise in an image while preserving edges

    num_iterations: how many times the filter is applied (bigger means more smoothing)

    kappa: threshold for diffusion coefficient (bigger means more diffusion across edges)

    gamma: integration constant for dffusion rate (keep <= 0.25 for stability)

    """
    for _ in range(num_iterations):
        # Gets directional gradients
        north = ndimage.shift(image, (-1, 0)) - image
        south = ndimage.shift(image, (1, 0)) - image
        east = ndimage.shift(image, (0, 1)) - image
        west = ndimage.shift(image, (0, -1)) - image

        # Gets directional Coefficients using exponential edge-stopping function
        c_north = np.exp(-((north / kappa) ** 2))
        c_south = np.exp(-((south / kappa) ** 2))
        c_east = np.exp(-((east / kappa) ** 2))
        c_west = np.exp(-((west / kappa) ** 2))

        # Updates image with the sum of the gradients
        image += gamma * (
            c_north * north + c_south * south + c_east * east + c_west * west
        )
    return image