import os
import cv2


def augment_single_image(image_path, label_path, output_dir, transform):

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with open(label_path, 'r') as f:
        lines = f.readlines()

    bboxes = []
    class_ids = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        class_id, xc, yc, w, h = map(float, parts)
        bboxes.append([xc, yc, w, h])
        class_ids.append(class_id)

    transformed = transform(image=image, bboxes=bboxes, class_ids=class_ids)
    new_image = transformed['image']
    new_bboxes = transformed.get('bboxes', [])
    new_class_ids = transformed.get('class_ids', [])

    images_output_dir = os.path.join(output_dir, "images")
    labels_output_dir = os.path.join(output_dir, "labels")

    output_image_path = os.path.join(images_output_dir, f"aug_{os.path.basename(image_path)}")
    cv2.imwrite(output_image_path, cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR))

    output_label_path = os.path.join(labels_output_dir, f"aug_{os.path.basename(label_path)}")
    with open(output_label_path, 'w') as f:
        for bbox, class_id in zip(new_bboxes, new_class_ids):
            f.write(f"{int(class_id)} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
    print(output_image_path, output_label_path)
