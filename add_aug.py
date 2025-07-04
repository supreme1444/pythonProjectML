
import albumentations as A

transform = A.Compose([

    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.1,
        rotate_limit=25,
        p=0.7
    ),

    A.OneOf([
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.8
        ),
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=40,
            val_shift_limit=30,
            p=0.8
        ),
    ], p=0.9),

    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 9), p=0.5),
        A.MotionBlur(blur_limit=11, p=0.3),
    ], p=0.5),

    A.Perspective(scale=(0.05, 0.1), p=0.3),

    A.LongestMaxSize(max_size=640),
    A.PadIfNeeded(
        min_height=640,
        min_width=640,
        border_mode=0,
    ),
], bbox_params=A.BboxParams(
    format='yolo',
    min_visibility=0.3,
    label_fields=['class_ids']
))
