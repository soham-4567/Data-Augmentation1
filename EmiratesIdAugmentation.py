import cv2
import albumentations as A
import os



def load_bboxes(file_path):
    bboxes = []
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            bboxes.append([x_center, y_center, width, height])
            labels.append(int(class_id))
    return bboxes, labels


def save_bboxes(file_path, bboxes, labels):
    with open(file_path, 'w') as file:
        for bbox, label in zip(bboxes, labels):
            x_center, y_center, width, height = bbox
            file.write(f"{label} {x_center} {y_center} {width} {height}\n")


augmentations = A.Compose([
    # A.RandomRotate90(p=1.0, always_apply=None),
    A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.8),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    A.RandomBrightnessContrast(p=0.8),
    A.HueSaturationValue(p=0.8),
    A.Blur(blur_limit=3, p=0.5),
    # A.RGBShift(p=1),
    A.ToGray(p=1),
    # A.ToSepia(p=1),
    A.RandomScale(scale_limit=0.2, p=0.8),
], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))


#images_dir = "C:/dataset/Emirates_id/Emid2/project-10-at-2024-08-19-16-07-4a143b46/images"
images_dir = "C:/Users/Intern/Downloads/ImageEmirateId"
# labels_dir = "C:/dataset/Emirates_id/Emid2/project-10-at-2024-08-19-16-07-4a143b46/labels"
labels_dir = "C:/Users/Intern/Downloads/LabelsEmirateId"
augmented_images_dir = 'augmented_images7'
augmented_labels_dir = 'augmented_labels7'

os.makedirs(augmented_images_dir, exist_ok=True)
os.makedirs(augmented_labels_dir, exist_ok=True)


for image_file in os.listdir(images_dir):
    if image_file.endswith('.jpg') or image_file.endswith('.png'):
        image_path = os.path.join(images_dir, image_file)
        label_path = os.path.join(labels_dir, os.path.splitext(image_file)[0] + '.txt')

        if not os.path.exists(label_path):
            print(f"Label file not found for image {image_file}, skipping...")
            continue

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes, labels = load_bboxes(label_path)



        for i in range(41, 51):
            augmented = augmentations(image=image, bboxes=bboxes, category_ids=labels)
            augmented_image = augmented['image']
            augmented_bboxes = augmented['bboxes']
            augmented_labels = augmented['category_ids']

            output_image_path = os.path.join(augmented_images_dir,
                                             f'{os.path.splitext(image_file)[0]}_augmented_{i}.jpg')
            cv2.imwrite(output_image_path, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))

            output_label_path = os.path.join(augmented_labels_dir,
                                             f'{os.path.splitext(image_file)[0]}_augmented_{i}.txt')
            save_bboxes(output_label_path, augmented_bboxes, augmented_labels)

print("Augmented images and labels saved successfully.")