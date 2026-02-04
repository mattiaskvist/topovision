"""Data augmentation pipelines for contour segmentation training."""

import albumentations
from albumentations.pytorch import ToTensorV2


def get_training_augmentations(image_size: int = 512) -> albumentations.Compose:
    """Get augmentation pipeline for training.

    Applies geometric and color augmentations to increase data diversity.
    The mask is automatically transformed alongside the image for geometric ops.

    Args:
        image_size: Target image size (square).

    Returns:
        Albumentations Compose pipeline.
    """
    return albumentations.Compose(
        [
            # Geometric augmentations
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.RandomRotate90(p=0.5),
            albumentations.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=45,
                border_mode=0,  # cv2.BORDER_CONSTANT
                value=0,
                mask_value=0,
                p=0.5,
            ),
            # Elastic deformation (useful for line detection)
            albumentations.ElasticTransform(
                alpha=50,
                sigma=10,
                border_mode=0,
                value=0,
                mask_value=0,
                p=0.3,
            ),
            # Color augmentations (image only, not mask)
            albumentations.OneOf(
                [
                    albumentations.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=1.0
                    ),
                    albumentations.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0
                    ),
                ],
                p=0.7,
            ),
            albumentations.OneOf(
                [
                    albumentations.GaussNoise(var_limit=(10, 50), p=1.0),
                    albumentations.ISONoise(
                        color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0
                    ),
                ],
                p=0.3,
            ),
            albumentations.OneOf(
                [
                    albumentations.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    albumentations.MotionBlur(blur_limit=(3, 5), p=1.0),
                ],
                p=0.2,
            ),
            # Ensure correct size
            albumentations.Resize(image_size, image_size),
            # Normalize and convert to tensor
            albumentations.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2(),
        ]
    )


def get_validation_augmentations(image_size: int = 512) -> albumentations.Compose:
    """Get augmentation pipeline for validation.

    Only applies normalization and tensor conversion, no augmentations.

    Args:
        image_size: Target image size (square).

    Returns:
        Albumentations Compose pipeline.
    """
    return albumentations.Compose(
        [
            albumentations.Resize(image_size, image_size),
            albumentations.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2(),
        ]
    )
