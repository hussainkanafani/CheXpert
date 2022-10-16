from imgaug import augmenters as iaa

flip_augmenter = iaa.Sequential(
    [
        iaa.Fliplr(0.1),
    ],
    random_order=True,
)

complex_augmenter = iaa.Sequential(
    [
        iaa.Fliplr(0.1),
        iaa.Affine(translate_px={"x": (-10, 10), "y": (-10, 10)}),
        iaa.Affine(rotate=(-2, 2)),
        iaa.Affine(scale=(0.9, 1.1))
    ],
    random_order=True,
)