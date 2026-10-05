import math
import os

import imageio
import wandb
from PIL import Image

import numpy as np


def save_image(id, image, log_wandb=True):
    """Saves an image in the [0,1]-valued Numpy array to image_path.

  Args:
    image: Numpy array of shape (height, width, {1,3}) with values in [0, 1].
    image_path: String with path to output image.
  """
    # Copy the single channel if we are provided a grayscale image.
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    image = np.ascontiguousarray(image)
    image *= 255.
    image = image.astype("uint8")

    img = Image.fromarray(image, mode="RGBA")

    if log_wandb:
        wandb.log({f"{id}": wandb.Image(img)})

    else:
        img.save(id, dpi=(1200, 1200))

def grid_save_images(id, images, log_wandb=True):
    """Saves images in list of [0,1]-valued np.arrays on a grid.

  Args:
    images: List of Numpy arrays of shape (height, width, {1,3}) with values in
      [0, 1].
    image_path: String with path to output image.
  """
    side_length = int(math.floor(math.sqrt(len(images))))
    image_rows = [
        np.concatenate(
            images[side_length * i:side_length * i + side_length], axis=0)
        for i in range(side_length)
    ]
    tiled_image = np.concatenate(image_rows, axis=1)
    save_image(id, tiled_image, log_wandb)


def pad_around(image, padding_px=10, axis=None, value=None):
    """Adds a padding around each image."""
    # If axis is None, pad both the first and the second axis.
    if axis is None:
        image = pad_around(image, padding_px, axis=0, value=value)
        axis = 1
    padding_arr = padding_array(image, padding_px, axis, value=value)
    return np.concatenate([padding_arr, image, padding_arr], axis=axis)


def save_animation(list_of_animated_images, image_path, fps):
    full_size_images = []
    for single_images in zip(*list_of_animated_images):
        full_size_images.append(
            pad_around(padded_grid(list(single_images))))

    full_size_images = np.array(full_size_images)
    imageio.mimwrite(image_path, full_size_images, fps=fps, format='gif')
    #wandb.log({f"traversal_{id}": wandb.Video(full_size_images, fps=fps, format="gif")})



def padding_array(image, padding_px, axis, value=None):
    """Creates padding image of proper shape to pad image along the axis."""
    shape = list(image.shape)
    shape[axis] = padding_px
    if value is None:
        return np.ones(shape, dtype=image.dtype)
    else:
        assert len(value) == shape[-1]
        shape[-1] = 1
        return np.tile(value, shape)


def best_num_rows(num_elements, max_ratio=4):
    """Automatically selects a smart number of rows."""
    best_remainder = num_elements
    best_i = None
    i = int(np.sqrt(num_elements))
    while True:
        if num_elements > max_ratio * i * i:
            return best_i
        remainder = (i - num_elements % i) % i
        if remainder == 0:
            return i
        if remainder < best_remainder:
            best_remainder = remainder
            best_i = i
        i -= 1


def padded_stack(images, padding_px=10, axis=0, value=None):
    """Stacks images along axis with padding in between images."""
    padding_arr = padding_array(images[0], padding_px, axis, value=value)
    new_images = [images[0]]
    for image in images[1:]:
        new_images.append(padding_arr)
        new_images.append(image)
    return np.concatenate(new_images, axis=axis)


def padded_grid(images, num_rows=None, padding_px=10, value=None):
    """Creates a grid with padding in between images."""
    num_images = len(images)
    if num_rows is None:
        num_rows = best_num_rows(num_images)

    # Computes how many empty images we need to add.
    num_cols = int(np.ceil(float(num_images) / num_rows))
    num_missing = num_rows * num_cols - num_images

    # Add the empty images at the end.
    all_images = images + [np.ones_like(images[0])] * num_missing

    # Create the final grid.
    rows = [padded_stack(all_images[i * num_cols:(i + 1) * num_cols], padding_px,
                         1, value=value) for i in range(num_rows)]
    return padded_stack(rows, padding_px, axis=0, value=value)



def traver_interval(starting_value, num_frames, min_val, max_val):
    """Cycles through the state space in a single cycle."""
    starting_in_01 = (starting_value - min_val) / (max_val - min_val)
    grid = np.linspace(starting_in_01, starting_in_01 + 2.,
                       num=num_frames, endpoint=False)
    grid -= np.maximum(0, 2 * grid - 2)
    grid += np.maximum(0, -2 * grid)
    return grid * (max_val - min_val) + min_val


