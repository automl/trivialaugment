# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Transforms used in the Augmentation Policies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
# pylint:disable=g-multiple-import
from PIL import ImageOps, ImageEnhance, ImageFilter, Image
# pylint:enable=g-multiple-import


IMAGE_SIZE = 32
# What is the dataset mean and std of the images on the training set
PARAMETER_MAX = 30  # What is the max 'level' a transform could be predicted

def pil_wrap(img):
  """Convert the `img` numpy tensor to a PIL Image."""
  return img.convert('RGBA')


def pil_unwrap(img):
  """Converts the PIL img to a numpy array."""
  return img.convert('RGB')

def apply_policy(policy, img, use_fixed_posterize=False):
  """Apply the `policy` to the numpy `img`.

  Args:
    policy: A list of tuples with the form (name, probability, level) where
      `name` is the name of the augmentation operation to apply, `probability`
      is the probability of applying the operation and `level` is what strength
      the operation to apply.
    img: Numpy image that will have `policy` applied to it.

  Returns:
    The result of applying `policy` to `img`.
  """
  nametotransform = fixed_AA_NAME_TO_TRANSFORM if use_fixed_posterize else AA_NAME_TO_TRANSFORM
  pil_img = pil_wrap(img)
  for xform in policy:
    assert len(xform) == 3
    name, probability, level = xform
    xform_fn = nametotransform[name].pil_transformer(probability, level)
    pil_img = xform_fn(pil_img)
  return pil_unwrap(pil_img)


def random_flip(x):
  """Flip the input x horizontally with 50% probability."""
  if np.random.rand(1)[0] > 0.5:
    return np.fliplr(x)
  return x


def zero_pad_and_crop(img, amount=4):
  """Zero pad by `amount` zero pixels on each side then take a random crop.

  Args:
    img: numpy image that will be zero padded and cropped.
    amount: amount of zeros to pad `img` with horizontally and verically.

  Returns:
    The cropped zero padded img. The returned numpy array will be of the same
    shape as `img`.
  """
  padded_img = np.zeros((img.shape[0] + amount * 2, img.shape[1] + amount * 2,
                         img.shape[2]))
  padded_img[amount:img.shape[0] + amount, amount:
             img.shape[1] + amount, :] = img
  top = np.random.randint(low=0, high=2 * amount)
  left = np.random.randint(low=0, high=2 * amount)
  new_img = padded_img[top:top + img.shape[0], left:left + img.shape[1], :]
  return new_img







def float_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.

  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
  return float(level) * maxval / PARAMETER_MAX


def int_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.

  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
  return int(level * maxval / PARAMETER_MAX)




class TransformFunction(object):
  """Wraps the Transform function for pretty printing options."""

  def __init__(self, func, name):
    self.f = func
    self.name = name

  def __repr__(self):
    return '<' + self.name + '>'

  def __call__(self, pil_img):
    return self.f(pil_img)


class TransformT(object):
  """Each instance of this class represents a specific transform."""

  def __init__(self, name, xform_fn):
    self.name = name
    self.xform = xform_fn

  def pil_transformer(self, probability, level):

    def return_function(im):
      if random.random() < probability:
        im = self.xform(im, level)
      return im

    name = self.name + '({:.1f},{})'.format(probability, level)
    return TransformFunction(return_function, name)

  def do_transform(self, image, level):
    f = self.pil_transformer(PARAMETER_MAX, level)
    return f(image)


################## Transform Functions ##################
identity = TransformT('identity', lambda pil_img, level: pil_img)
flip_lr = TransformT(
    'FlipLR',
    lambda pil_img, level: pil_img.transpose(Image.FLIP_LEFT_RIGHT))
flip_ud = TransformT(
    'FlipUD',
    lambda pil_img, level: pil_img.transpose(Image.FLIP_TOP_BOTTOM))
# pylint:disable=g-long-lambda
auto_contrast = TransformT(
    'AutoContrast',
    lambda pil_img, level: ImageOps.autocontrast(
        pil_img.convert('RGB')).convert('RGBA'))
equalize = TransformT(
    'Equalize',
    lambda pil_img, level: ImageOps.equalize(
        pil_img.convert('RGB')).convert('RGBA'))
invert = TransformT(
    'Invert',
    lambda pil_img, level: ImageOps.invert(
        pil_img.convert('RGB')).convert('RGBA'))
# pylint:enable=g-long-lambda
blur = TransformT(
    'Blur', lambda pil_img, level: pil_img.filter(ImageFilter.BLUR))
smooth = TransformT(
    'Smooth',
    lambda pil_img, level: pil_img.filter(ImageFilter.SMOOTH))


def _rotate_impl(pil_img, level):
  """Rotates `pil_img` from -30 to 30 degrees depending on `level`."""
  degrees = int_parameter(level, 30)
  if random.random() > 0.5:
    degrees = -degrees
  return pil_img.rotate(degrees)


rotate = TransformT('Rotate', _rotate_impl)


def _posterize_impl(pil_img, level):
  """Applies PIL Posterize to `pil_img`."""
  level = int_parameter(level, 4)
  return ImageOps.posterize(pil_img.convert('RGB'), 4 - level).convert('RGBA')


posterize = TransformT('Posterize', _posterize_impl)

def _fixed_posterize_impl(pil_img, level):
  """Applies PIL Posterize to `pil_img`."""
  level = int_parameter(level, 4)
  return ImageOps.posterize(pil_img.convert('RGB'), 8 - level).convert('RGBA')

fixed_posterize = TransformT('Posterize', _fixed_posterize_impl)


def _shear_x_impl(pil_img, level):
  """Applies PIL ShearX to `pil_img`.

  The ShearX operation shears the image along the horizontal axis with `level`
  magnitude.

  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].

  Returns:
    A PIL Image that has had ShearX applied to it.
  """
  level = float_parameter(level, 0.3)
  if random.random() > 0.5:
    level = -level
  return pil_img.transform((32, 32), Image.AFFINE, (1, level, 0, 0, 1, 0))


shear_x = TransformT('ShearX', _shear_x_impl)


def _shear_y_impl(pil_img, level):
  """Applies PIL ShearY to `pil_img`.

  The ShearY operation shears the image along the vertical axis with `level`
  magnitude.

  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].

  Returns:
    A PIL Image that has had ShearX applied to it.
  """
  level = float_parameter(level, 0.3)
  if random.random() > 0.5:
    level = -level
  return pil_img.transform((32, 32), Image.AFFINE, (1, 0, 0, level, 1, 0))


shear_y = TransformT('ShearY', _shear_y_impl)


def _translate_x_impl(pil_img, level):
  """Applies PIL TranslateX to `pil_img`.

  Translate the image in the horizontal direction by `level`
  number of pixels.

  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].

  Returns:
    A PIL Image that has had TranslateX applied to it.
  """
  level = int_parameter(level, 10)
  if random.random() > 0.5:
    level = -level
  return pil_img.transform((32, 32), Image.AFFINE, (1, 0, level, 0, 1, 0))


translate_x = TransformT('TranslateX', _translate_x_impl)


def _translate_y_impl(pil_img, level):
  """Applies PIL TranslateY to `pil_img`.

  Translate the image in the vertical direction by `level`
  number of pixels.

  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].

  Returns:
    A PIL Image that has had TranslateY applied to it.
  """
  level = int_parameter(level, 10)
  if random.random() > 0.5:
    level = -level
  return pil_img.transform((32, 32), Image.AFFINE, (1, 0, 0, 0, 1, level))


translate_y = TransformT('TranslateY', _translate_y_impl)


def _crop_impl(pil_img, level, interpolation=Image.BILINEAR):
  """Applies a crop to `pil_img` with the size depending on the `level`."""
  cropped = pil_img.crop((level, level, IMAGE_SIZE - level, IMAGE_SIZE - level))
  resized = cropped.resize((IMAGE_SIZE, IMAGE_SIZE), interpolation)
  return resized


crop_bilinear = TransformT('CropBilinear', _crop_impl)


def _solarize_impl(pil_img, level):
  """Applies PIL Solarize to `pil_img`.

  Translate the image in the vertical direction by `level`
  number of pixels.

  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].

  Returns:
    A PIL Image that has had Solarize applied to it.
  """
  level = int_parameter(level, 256)
  return ImageOps.solarize(pil_img.convert('RGB'), 256 - level).convert('RGBA')


solarize = TransformT('Solarize', _solarize_impl)


def _enhancer_impl(enhancer):
  """Sets level to be between 0.1 and 1.8 for ImageEnhance transforms of PIL."""
  def impl(pil_img, level):
    v = float_parameter(level, 1.8) + .1  # going to 0 just destroys it
    return enhancer(pil_img).enhance(v)
  return impl


color = TransformT('Color', _enhancer_impl(ImageEnhance.Color))
contrast = TransformT('Contrast', _enhancer_impl(ImageEnhance.Contrast))
brightness = TransformT('Brightness', _enhancer_impl(
    ImageEnhance.Brightness))
sharpness = TransformT('Sharpness', _enhancer_impl(ImageEnhance.Sharpness))

def create_cutout_mask(img_height, img_width, num_channels, size):
  """Creates a zero mask used for cutout of shape `img_height` x `img_width`.

  Args:
    img_height: Height of image cutout mask will be applied to.
    img_width: Width of image cutout mask will be applied to.
    num_channels: Number of channels in the image.
    size: Size of the zeros mask.

  Returns:
    A mask of shape `img_height` x `img_width` with all ones except for a
    square of zeros of shape `size` x `size`. This mask is meant to be
    elementwise multiplied with the original image. Additionally returns
    the `upper_coord` and `lower_coord` which specify where the cutout mask
    will be applied.
  """
  assert img_height == img_width

  # Sample center where cutout mask will be applied
  height_loc = np.random.randint(low=0, high=img_height)
  width_loc = np.random.randint(low=0, high=img_width)

  # Determine upper right and lower left corners of patch
  upper_coord = (max(0, height_loc - size // 2), max(0, width_loc - size // 2))
  lower_coord = (min(img_height, height_loc + size // 2),
                 min(img_width, width_loc + size // 2))
  mask_height = lower_coord[0] - upper_coord[0]
  mask_width = lower_coord[1] - upper_coord[1]
  assert mask_height > 0
  assert mask_width > 0

  mask = np.ones((img_height, img_width, num_channels))
  zeros = np.zeros((mask_height, mask_width, num_channels))
  mask[upper_coord[0]:lower_coord[0], upper_coord[1]:lower_coord[1], :] = (
      zeros)
  return mask, upper_coord, lower_coord

def _cutout_pil_impl(pil_img, level):
  """Apply cutout to pil_img at the specified level."""
  size = int_parameter(level, 20)
  if size <= 0:
    return pil_img
  img_height, img_width, num_channels = (32, 32, 3)
  _, upper_coord, lower_coord = (
      create_cutout_mask(img_height, img_width, num_channels, size))
  pixels = pil_img.load()  # create the pixel map
  for i in range(upper_coord[0], lower_coord[0]):  # for every col:
    for j in range(upper_coord[1], lower_coord[1]):  # For every row
      pixels[i, j] = (125, 122, 113, 0)  # set the colour accordingly
  return pil_img

cutout = TransformT('Cutout', _cutout_pil_impl)



ALL_TRANSFORMS = [
    identity,
    auto_contrast,
    equalize,
    rotate,
    posterize,
    solarize,
    color,
    contrast,
    brightness,
    sharpness,
    shear_x,
    shear_y,
    translate_x,
    translate_y,
]

AA_ALL_TRANSFORMS = [
    flip_lr,
    flip_ud,
    auto_contrast,
    equalize,
    invert,
    rotate,
    posterize,
    crop_bilinear,
    solarize,
    color,
    contrast,
    brightness,
    sharpness,
    shear_x,
    shear_y,
    translate_x,
    translate_y,
    cutout,
    blur,
    smooth
]


fixed_AA_ALL_TRANSFORMS = [
    flip_lr,
    flip_ud,
    auto_contrast,
    equalize,
    invert,
    rotate,
    fixed_posterize,
    crop_bilinear,
    solarize,
    color,
    contrast,
    brightness,
    sharpness,
    shear_x,
    shear_y,
    translate_x,
    translate_y,
    cutout,
    blur,
    smooth
]


class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m      # [0, 30]

    def __call__(self, img):
        img = pil_wrap(img)
        ops = random.choices(ALL_TRANSFORMS, k=self.n)
        for op in ops:
            img = op.pil_transformer(1.,self.m)(img)
        img = pil_unwrap(img)

        return img

AA_NAME_TO_TRANSFORM = {t.name: t for t in AA_ALL_TRANSFORMS}
fixed_AA_NAME_TO_TRANSFORM = {t.name: t for t in fixed_AA_ALL_TRANSFORMS}

NAME_TO_TRANSFORM = {t.name: t for t in ALL_TRANSFORMS}

def good_policies():
  """AutoAugment policies found on Cifar."""
  exp0_0 = [
      [('Invert', 0.1, 7), ('Contrast', 0.2, 6)],
      [('Rotate', 0.7, 2), ('TranslateX', 0.3, 9)],
      [('Sharpness', 0.8, 1), ('Sharpness', 0.9, 3)],
      [('ShearY', 0.5, 8), ('TranslateY', 0.7, 9)],
      [('AutoContrast', 0.5, 8), ('Equalize', 0.9, 2)]]
  exp0_1 = [
      [('Solarize', 0.4, 5), ('AutoContrast', 0.9, 3)],
      [('TranslateY', 0.9, 9), ('TranslateY', 0.7, 9)],
      [('AutoContrast', 0.9, 2), ('Solarize', 0.8, 3)],
      [('Equalize', 0.8, 8), ('Invert', 0.1, 3)],
      [('TranslateY', 0.7, 9), ('AutoContrast', 0.9, 1)]]
  exp0_2 = [
      [('Solarize', 0.4, 5), ('AutoContrast', 0.0, 2)],
      [('TranslateY', 0.7, 9), ('TranslateY', 0.7, 9)],
      [('AutoContrast', 0.9, 0), ('Solarize', 0.4, 3)],
      [('Equalize', 0.7, 5), ('Invert', 0.1, 3)],
      [('TranslateY', 0.7, 9), ('TranslateY', 0.7, 9)]]
  exp0_3 = [
      [('Solarize', 0.4, 5), ('AutoContrast', 0.9, 1)],
      [('TranslateY', 0.8, 9), ('TranslateY', 0.9, 9)],
      [('AutoContrast', 0.8, 0), ('TranslateY', 0.7, 9)],
      [('TranslateY', 0.2, 7), ('Color', 0.9, 6)],
      [('Equalize', 0.7, 6), ('Color', 0.4, 9)]]
  exp1_0 = [
      [('ShearY', 0.2, 7), ('Posterize', 0.3, 7)],
      [('Color', 0.4, 3), ('Brightness', 0.6, 7)],
      [('Sharpness', 0.3, 9), ('Brightness', 0.7, 9)],
      [('Equalize', 0.6, 5), ('Equalize', 0.5, 1)],
      [('Contrast', 0.6, 7), ('Sharpness', 0.6, 5)]]
  exp1_1 = [
      [('Brightness', 0.3, 7), ('AutoContrast', 0.5, 8)],
      [('AutoContrast', 0.9, 4), ('AutoContrast', 0.5, 6)],
      [('Solarize', 0.3, 5), ('Equalize', 0.6, 5)],
      [('TranslateY', 0.2, 4), ('Sharpness', 0.3, 3)],
      [('Brightness', 0.0, 8), ('Color', 0.8, 8)]]
  exp1_2 = [
      [('Solarize', 0.2, 6), ('Color', 0.8, 6)],
      [('Solarize', 0.2, 6), ('AutoContrast', 0.8, 1)],
      [('Solarize', 0.4, 1), ('Equalize', 0.6, 5)],
      [('Brightness', 0.0, 0), ('Solarize', 0.5, 2)],
      [('AutoContrast', 0.9, 5), ('Brightness', 0.5, 3)]]
  exp1_3 = [
      [('Contrast', 0.7, 5), ('Brightness', 0.0, 2)],
      [('Solarize', 0.2, 8), ('Solarize', 0.1, 5)],
      [('Contrast', 0.5, 1), ('TranslateY', 0.2, 9)],
      [('AutoContrast', 0.6, 5), ('TranslateY', 0.0, 9)],
      [('AutoContrast', 0.9, 4), ('Equalize', 0.8, 4)]]
  exp1_4 = [
      [('Brightness', 0.0, 7), ('Equalize', 0.4, 7)],
      [('Solarize', 0.2, 5), ('Equalize', 0.7, 5)],
      [('Equalize', 0.6, 8), ('Color', 0.6, 2)],
      [('Color', 0.3, 7), ('Color', 0.2, 4)],
      [('AutoContrast', 0.5, 2), ('Solarize', 0.7, 2)]]
  exp1_5 = [
      [('AutoContrast', 0.2, 0), ('Equalize', 0.1, 0)],
      [('ShearY', 0.6, 5), ('Equalize', 0.6, 5)],
      [('Brightness', 0.9, 3), ('AutoContrast', 0.4, 1)],
      [('Equalize', 0.8, 8), ('Equalize', 0.7, 7)],
      [('Equalize', 0.7, 7), ('Solarize', 0.5, 0)]]
  exp1_6 = [
      [('Equalize', 0.8, 4), ('TranslateY', 0.8, 9)],
      [('TranslateY', 0.8, 9), ('TranslateY', 0.6, 9)],
      [('TranslateY', 0.9, 0), ('TranslateY', 0.5, 9)],
      [('AutoContrast', 0.5, 3), ('Solarize', 0.3, 4)],
      [('Solarize', 0.5, 3), ('Equalize', 0.4, 4)]]
  exp2_0 = [
      [('Color', 0.7, 7), ('TranslateX', 0.5, 8)],
      [('Equalize', 0.3, 7), ('AutoContrast', 0.4, 8)],
      [('TranslateY', 0.4, 3), ('Sharpness', 0.2, 6)],
      [('Brightness', 0.9, 6), ('Color', 0.2, 8)],
      [('Solarize', 0.5, 2), ('Invert', 0.0, 3)]]
  exp2_1 = [
      [('AutoContrast', 0.1, 5), ('Brightness', 0.0, 0)],
      [('Cutout', 0.2, 4), ('Equalize', 0.1, 1)],
      [('Equalize', 0.7, 7), ('AutoContrast', 0.6, 4)],
      [('Color', 0.1, 8), ('ShearY', 0.2, 3)],
      [('ShearY', 0.4, 2), ('Rotate', 0.7, 0)]]
  exp2_2 = [
      [('ShearY', 0.1, 3), ('AutoContrast', 0.9, 5)],
      [('TranslateY', 0.3, 6), ('Cutout', 0.3, 3)],
      [('Equalize', 0.5, 0), ('Solarize', 0.6, 6)],
      [('AutoContrast', 0.3, 5), ('Rotate', 0.2, 7)],
      [('Equalize', 0.8, 2), ('Invert', 0.4, 0)]]
  exp2_3 = [
      [('Equalize', 0.9, 5), ('Color', 0.7, 0)],
      [('Equalize', 0.1, 1), ('ShearY', 0.1, 3)],
      [('AutoContrast', 0.7, 3), ('Equalize', 0.7, 0)],
      [('Brightness', 0.5, 1), ('Contrast', 0.1, 7)],
      [('Contrast', 0.1, 4), ('Solarize', 0.6, 5)]]
  exp2_4 = [
      [('Solarize', 0.2, 3), ('ShearX', 0.0, 0)],
      [('TranslateX', 0.3, 0), ('TranslateX', 0.6, 0)],
      [('Equalize', 0.5, 9), ('TranslateY', 0.6, 7)],
      [('ShearX', 0.1, 0), ('Sharpness', 0.5, 1)],
      [('Equalize', 0.8, 6), ('Invert', 0.3, 6)]]
  exp2_5 = [
      [('AutoContrast', 0.3, 9), ('Cutout', 0.5, 3)],
      [('ShearX', 0.4, 4), ('AutoContrast', 0.9, 2)],
      [('ShearX', 0.0, 3), ('Posterize', 0.0, 3)],
      [('Solarize', 0.4, 3), ('Color', 0.2, 4)],
      [('Equalize', 0.1, 4), ('Equalize', 0.7, 6)]]
  exp2_6 = [
      [('Equalize', 0.3, 8), ('AutoContrast', 0.4, 3)],
      [('Solarize', 0.6, 4), ('AutoContrast', 0.7, 6)],
      [('AutoContrast', 0.2, 9), ('Brightness', 0.4, 8)],
      [('Equalize', 0.1, 0), ('Equalize', 0.0, 6)],
      [('Equalize', 0.8, 4), ('Equalize', 0.0, 4)]]
  exp2_7 = [
      [('Equalize', 0.5, 5), ('AutoContrast', 0.1, 2)],
      [('Solarize', 0.5, 5), ('AutoContrast', 0.9, 5)],
      [('AutoContrast', 0.6, 1), ('AutoContrast', 0.7, 8)],
      [('Equalize', 0.2, 0), ('AutoContrast', 0.1, 2)],
      [('Equalize', 0.6, 9), ('Equalize', 0.4, 4)]]
  exp0s = exp0_0 + exp0_1 + exp0_2 + exp0_3
  exp1s = exp1_0 + exp1_1 + exp1_2 + exp1_3 + exp1_4 + exp1_5 + exp1_6
  exp2s = exp2_0 + exp2_1 + exp2_2 + exp2_3 + exp2_4 + exp2_5 + exp2_6 + exp2_7
  return  exp0s + exp1s + exp2s

cifar_gp = good_policies()

first_aug_ops = [("ShearX",0.9,4), ("ShearY",0.9,8), ("Equalize",0.6,5), ("Invert",0.9,3), ("Equalize",0.6,1), ("ShearX",0.9,4), ("ShearY",0.9,8), ("ShearY",0.9,5), ("Invert",0.9,6), ("Equalize",0.6,3), ("ShearX",0.9,4), ("ShearY",0.8,8), ("Equalize",0.9,5), ("Invert",0.9,4), ("Contrast",0.3,3), ("Invert",0.8,5), ("ShearY",0.7,6), ("Invert",0.6,4), ("ShearY",0.3,7), ("ShearX",0.1,6), ("Solarize",0.7,2), ("ShearY",0.8,4), ("ShearX",0.7,9), ("ShearY",0.8,5), ("ShearX",0.7,2)]
second_aug_ops = [("Invert",0.2,3), ("Invert",0.7,5), ("Solarize",0.6,6), ("Equalize",0.6,3), ("Rotate",0.9,3), ("AutoContrast",0.8,3), ("Invert",0.4,5), ("Solarize",0.2,6), ("AutoContrast",0.8,1), ("Rotate",0.9,3), ("Solarize",0.3,3), ("Invert",0.7,4), ("TranslateY",0.6,6), ("Equalize",0.6,7), ("Rotate",0.8,4), ("TranslateY",0.0,2), ("Solarize",0.4,8), ("Rotate",0.8,4), ("TranslateX",0.9,3), ("Invert",0.6,5), ("TranslateY",0.6,7), ("Invert",0.8,8), ("TranslateY",0.8,3), ("AutoContrast",0.7,3), ("Invert",0.1,5)]

svhn_gp = [[a1, a2] for a1, a2 in zip(first_aug_ops,second_aug_ops)]

class CifarAutoAugment:
    def __init__(self, fixed_posterize):
        self.fixed_posterize = fixed_posterize

    def __call__(self, img):
        epoch_policy = cifar_gp[np.random.choice(len(cifar_gp))]
        final_img = apply_policy(epoch_policy, img, use_fixed_posterize=self.fixed_posterize)

        return final_img

class SVHNAutoAugment:
    def __init__(self, fixed_posterize):
        self.fixed_posterize = fixed_posterize

    def __call__(self, img):
        epoch_policy = svhn_gp[np.random.choice(len(svhn_gp))]
        final_img = apply_policy(epoch_policy, img, use_fixed_posterize=self.fixed_posterize)

        return final_img
