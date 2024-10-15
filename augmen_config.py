
import json
import albumentations as A
import cv2
import os


# Function to dynamically load augmentations and loops from JSON config
def load_augmentations_and_loops_from_config(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)

    # List to store the augmentations and their loop count
    augmentations_list = []
    augmentation_loops = []

    # Loop over augmentations in config
    for aug in config["augmentations"]:
        aug_name = aug["name"]
        aug_params = aug["params"]
        aug_loops = aug.get("loops", 1)  # Default to 1 loop if not specified

        # Dynamically get the augmentation class from the albumentations module
        augmentation = getattr(A, aug_name)(**aug_params)
        augmentations_list.append(augmentation)
        augmentation_loops.append(aug_loops)

    # Get bbox parameters
    bbox_params = A.BboxParams(**config["bbox_params"])

    # Return augmentations, their loops, and bbox_params
    return augmentations_list, augmentation_loops, bbox_params


# Function to load bounding boxes from a file
def load_bboxes(file_path):
    bboxes = []
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            bboxes.append([x_center, y_center, width, height])
            labels.append(int(class_id))
    return bboxes, labels


# Function to save bounding boxes to a file
def save_bboxes(file_path, bboxes, labels):
    with open(file_path, 'w') as file:
        for bbox, label in zip(bboxes, labels):
            x_center, y_center, width, height = bbox
            file.write(f"{label} {x_center} {y_center} {width} {height}\n")


# Load augmentations and loops from JSON config file
config_file = 'augmentations_config.json'
augmentations, augmentation_loops, bbox_params = load_augmentations_and_loops_from_config(config_file)

# Paths for images and labels
images_dir = "C:/Users/Intern/Downloads/ImageEmirateId"
labels_dir = "C:/Users/Intern/Downloads/LabelsEmirateId"
augmented_images_dir = 'augmented_images7'
augmented_labels_dir = 'augmented_labels7'

# Create directories for augmented images and labels if they don't exist
os.makedirs(augmented_images_dir, exist_ok=True)
os.makedirs(augmented_labels_dir, exist_ok=True)

# Loop through the images directory
for image_file in os.listdir(images_dir):
    if image_file.endswith('.jpg') or image_file.endswith('.png'):
        image_path = os.path.join(images_dir, image_file)
        label_path = os.path.join(labels_dir, os.path.splitext(image_file)[0] + '.txt')

        # If the label file does not exist, skip the image
        if not os.path.exists(label_path):
            print(f"Label file not found for image {image_file}, skipping...")
            continue

        # Load the image and bounding boxes
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes, labels = load_bboxes(label_path)

        # Apply each augmentation the specified number of times (loops)
        for aug, loops in zip(augmentations, augmentation_loops):
            for loop in range(loops):
                augmented = A.Compose([aug], bbox_params=bbox_params)(image=image, bboxes=bboxes, category_ids=labels)
                augmented_image = augmented['image']
                augmented_bboxes = augmented['bboxes']
                augmented_labels = augmented['category_ids']

                # Save the augmented image and bounding boxes with unique filenames
                output_image_path = os.path.join(augmented_images_dir,
                                                 f'{os.path.splitext(image_file)[0]}augmented{aug.__class__.__name__}{loop + 1}.jpg')
                cv2.imwrite(output_image_path, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))

                output_label_path = os.path.join(augmented_labels_dir,
                                                 f'{os.path.splitext(image_file)[0]}augmented{aug.__class__.__name__}{loop + 1}.txt')
                save_bboxes(output_label_path, augmented_bboxes, augmented_labels)

print("Augmented images and labels saved successfully.")
