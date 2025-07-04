import os
from add_aug import transform
from get_image import augment_single_image


def open_test():
    images_dir = "dataset_base_pic/images"
    labels_dir = "dataset_base_pic/labels"
    output_dir = "augmented"

    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

    for image_file in image_files:
        frame_name = os.path.splitext(image_file)[0]

        image_path = os.path.join(images_dir, image_file)
        label_path = os.path.join(labels_dir, f"{frame_name}.txt")
        if os.path.exists(label_path):
            augment_single_image(image_path, label_path, output_dir, transform, )


if __name__ == "__main__":
    open_test()
