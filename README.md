# Trivial Augment

This is the official implementation of TrivialAugment, as was used for the paper.
We distribute this implementation with two main use cases in mind.
Either you only use our (re-)implementetations of practical augmentation methods or you start off with our full codebase.

## Use TrivialAugment and Other Methods in Your Own Codebase
In this case we recommend to simply copy over the file `aug_lib.py` to your codebase.
You can now instantiate the augmenters `TrivialAugment`, `RandAugment` and `UniAugment` like this:
```
augmenter = aug_lib.TrivialAugment()
```
And simply use them on a PIL images `img`:
```
aug_img = augmenter(img)
```
This format also happens to be compatible with `torchvision.transforms`.
If you do not have `Pillow` or `numpy` installed, do so by calling `pip install Pillow numpy`.
Generally, a good position to augment an image with the `augmenter` is right as you get it out of the dataset, before you apply any custom augmentations.

The default augmentation space is `fixed_standard`, that is without AutoAugments posterization bug and using the set of augmentations used in Randaugment.
This is the search space we used for all our experiments, that do not mention another augmentation space.
You can change the augmentation space, though, with `aug_lib.set_augmentation_space`.
This call for example
```
aug_lib.set_augmentation_space('fixed_custom',2,['cutout'])
```
will change the augmentation space to only ever apply cutout with a large width or nothing.
The 2 here gives indications in how many strength levels the strength ranges of the augmentation space should be divided.
If an augmentation space includes `sample_pairing`, you need to specify a set of images with which to pair before each step:
`aug_lib.blend_images = [LIST OF PIL IMAGES]`.

## Use Our Full Codebase
Clone this directory and `cd` into it.
```
git clone ANONYMIZED/trivialaugment
cd trivialaugment
```
Install a fitting PyTorch version for your setup with GPU support,
as our implementation only support setups with at least one CUDA device and
install our requirements:
```
pip install -r requirements.txt
# Install a pytorch version, in many setups this has to be done manually, see pytorch.org
```

Now you should be ready to go. Start a training like so:
```
python -m TrivialAugment.train -c confs/wresnet40x2_cifar100_b128_maxlr.1_ta_fixedsesp_nowarmup_200epochs.yaml --dataroot data --tag EXPERIMENT_NAME
```
For concrete configs of experiments from the paper see the comments in the papers LaTeX code around the number you want to reproduce.
For logs and metrics use a `tensorboard` with the `logs` directory or use our `aggregate_results.py` script to view data from the `tensorboard` logs in the command line.

## Confidence Intervals
Since in the current literature we rarely found confidence intervals, we share our implementation in `evaluation_tools.py`.

