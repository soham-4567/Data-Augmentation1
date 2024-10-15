# import cv2
# import random
# import os
# import matplotlib.pyplot as plt
#
# # Function to load bounding boxes from a file
# def load_bboxes(file_path):
#     bboxes = []
#     labels = []
#     with open(file_path, 'r') as file:
#         for line in file:
#             class_id, x_center, y_center, width, height = map(float, line.strip().split())
#             bboxes.append([x_center, y_center, width, height])
#             labels.append(int(class_id))
#     return bboxes, labels
#
# # Function to plot the image with bounding boxes
# def plot_image_with_boxes(image, bboxes):
#     # Plot the image using matplotlib
#     plt.figure(figsize=(10, 10))
#     plt.imshow(image)
#
#     # Get image dimensions (for converting YOLO bbox format to pixel values)
#     img_h, img_w, _ = image.shape
#
#     # Loop over all bounding boxes and plot them
#     for bbox in bboxes:
#         x_center, y_center, width, height = bbox
#
#         # Convert from YOLO format (x_center, y_center, width, height) to (x_min, y_min, x_max, y_max)
#         x_min = int((x_center - width / 2) * img_w)
#         y_min = int((y_center - height / 2) * img_h)
#         x_max = int((x_center + width / 2) * img_w)
#         y_max = int((y_center + height / 2) * img_h)
#
#         # Draw the bounding box
#         plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
#                                           edgecolor='red', facecolor='none', linewidth=2))
#
#     plt.show()
#
# # Function to randomly pick an image and corresponding label, then plot bounding boxes
# def pick_and_plot_random_image(augmented_images_dir, augmented_labels_dir,n):
#
#     # Get list of all augmented image files
#     image_files = [f for f in os.listdir(augmented_images_dir) if f.endswith('.jpg') or f.endswith('.png')]
#
#     # Randomly select an image
#     selected_image_file = random.sample(image_files,n)
#
#     for i in range(n):
#         image_base_name =[]
#
#         # Get corresponding label file
#         image_base_name.append(os.path.splitext(selected_image_file[i]))
#         label_file=[]
#         label_file.append(os.path.join(augmented_labels_dir, image_base_name[i] + '.txt'))
#
#         # Check if the corresponding label file exists
#         # if not os.path.exists(label_file):
#         #     print(f"Label file {label_file} not found.")
#         #     return
#
#     # Load image
#     image_path = os.path.join(augmented_images_dir, selected_image_file)
#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     for i in range(n):
#
#         # Load bounding boxes and labels
#         bboxes, _ =[]
#         bboxes, _.append(load_bboxes(label_file[i]))  # Reuse the load_bboxes function from earlier
#
#         # Plot the image with bounding boxes
#         plot_image_with_boxes(image, bboxes[i])
#
# # Directory paths (set according to your setup)
# augmented_images_dir = 'augmented_images7'
# augmented_labels_dir = 'augmented_labels7'
#
# n=int(input("enter the value of n"))
#
# # Randomly pick and plot an image with bounding boxes
# pick_and_plot_random_image(augmented_images_dir, augmented_labels_dir,n)


# import cv2
# import random
# import os
# import matplotlib.pyplot as plt
#
# # Function to load bounding boxes from a file
# def load_bboxes(file_path):
#     bboxes = []
#     labels = []
#     with open(file_path, 'r') as file:
#         for line in file:
#             class_id, x_center, y_center, width, height = map(float, line.strip().split())
#             bboxes.append([x_center, y_center, width, height])
#             labels.append(int(class_id))
#     return bboxes, labels
#
# # Function to plot the image with bounding boxes
# def plot_image_with_boxes(image, bboxes):
#     plt.figure(figsize=(10, 10))
#     plt.imshow(image)
#     img_h, img_w, _ = image.shape
#
#     for bbox in bboxes:
#         x_center, y_center, width, height = bbox
#         x_min = int((x_center - width / 2) * img_w)
#         y_min = int((y_center - height / 2) * img_h)
#         x_max = int((x_center + width / 2) * img_w)
#         y_max = int((y_center + height / 2) * img_h)
#         plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
#                                           edgecolor='red', facecolor='none', linewidth=2))
#
#     plt.axis('on')  # Hide axis
#     plt.show()
#
# # Function to randomly pick images and plot them
# def pick_and_plot_random_images(augmented_images_dir, augmented_labels_dir, num_images):
#     image_files = [f for f in os.listdir(augmented_images_dir) if f.endswith('.jpg') or f.endswith('.png')]
#
#     if num_images > len(image_files):
#         print(f"Number of requested images ({num_images}) exceeds available images ({len(image_files)}).")
#         num_images = len(image_files)  # Adjust to available images if necessary
#
#     selected_images = random.sample(image_files, num_images)  # Randomly select the specified number of images
#
#     for image_file in selected_images:
#         image_path = os.path.join(augmented_images_dir, image_file)
#         label_file = os.path.join(augmented_labels_dir, os.path.splitext(image_file)[0] + '.txt')
#
#         if not os.path.exists(label_file):
#             print(f"Label file not found for image {image_file}, skipping...")
#             continue
#
#         image = cv2.imread(image_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         bboxes, _ = load_bboxes(label_file)  # Reuse the load_bboxes function
#
#         plot_image_with_boxes(image, bboxes)
#
# # Get user input for number of images to select
# try:
#     num_images = int(input("Enter the number of images to randomly select and plot: "))
# except ValueError:
#     print("Invalid input. Please enter a valid integer.")
#     num_images = 0  # Set to 0 to exit
#
# # Directory paths (set according to your setup)
# augmented_images_dir = 'augmented_images7'
# augmented_labels_dir = 'augmented_labels7'
#
# # Pick and plot the specified number of images
# if num_images > 0:
#     pick_and_plot_random_images(augmented_images_dir, augmented_labels_dir, num_images)



import cv2
import random
import numpy as np
import os
import matplotlib.pyplot as plt
import albumentations as A

# Define functions to handle bounding boxes and image rotation

def rotate_point(x, y, angle, cx, cy):
    """Rotate a point around a center (cx, cy) by an angle."""
    angle_rad = np.deg2rad(angle)
    x -= cx
    y -= cy
    x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
    y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad)
    return x_rot + cx, y_rot + cy

def rotate_bboxes(bboxes, angle, img_w, img_h):
    """Rotate bounding boxes by angle."""
    new_bboxes = []
    for bbox in bboxes:
        x_center, y_center, width, height = bbox
        x_center *= img_w
        y_center *= img_h
        width *= img_w
        height *= img_h

        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2

        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2

        corners = [
            (x_min, y_min),
            (x_max, y_min),
            (x_max, y_max),
            (x_min, y_max)
        ]

        rotated_corners = [rotate_point(x, y, angle, cx, cy) for x, y in corners]
        rotated_corners = np.array(rotated_corners)
        x_min_rot = np.min(rotated_corners[:, 0])
        y_min_rot = np.min(rotated_corners[:, 1])
        x_max_rot = np.max(rotated_corners[:, 0])
        y_max_rot = np.max(rotated_corners[:, 1])

        x_center_rot = (x_min_rot + x_max_rot) / 2 / img_w
        y_center_rot = (y_min_rot + y_max_rot) / 2 / img_h
        width_rot = (x_max_rot - x_min_rot) / img_w
        height_rot = (y_max_rot - y_min_rot) / img_h

        new_bboxes.append([x_center_rot, y_center_rot, width_rot, height_rot])

    return new_bboxes

def load_bboxes(file_path):
    bboxes = []
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            bboxes.append([x_center, y_center, width, height])
            labels.append(int(class_id))
    return bboxes, labels

def save_bboxes(file_path, new_bboxes, labels):
    with open(file_path, 'w') as file:
        for bbox, label in zip(new_bboxes, labels):
            x_center, y_center, width, height = bbox
            file.write(f"{label} {x_center} {y_center} {width} {height}\n")

def plot_image_with_boxes(image, new_bboxes):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    img_h, img_w, _ = image.shape
    for bbox in new_bboxes:
        x_center, y_center, width, height = bbox
        x_min = int((x_center - width / 2) * img_w)
        y_min = int((y_center - height / 2) * img_h)
        x_max = int((x_center + width / 2) * img_w)
        y_max = int((y_center + height / 2) * img_h)
        plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                          edgecolor='red', facecolor='none', linewidth=2))
    plt.axis('off')
    plt.show()

def apply_augmentations_and_plot(image, new_bboxes, labels):
    # Apply augmentation
    augmented = augmentations(image=image, new_bboxes=new_bboxes, category_ids=labels)
    augmented_image = augmented['image']
    augmented_bboxes = augmented['new_bboxes']
    augmented_labels = augmented['category_ids']

    # Plot the augmented image with bounding boxes
    plot_image_with_boxes(augmented_image, augmented_bboxes)

def pick_and_plot_random_images(augmented_images_dir, augmented_labels_dir, num_images):
    image_files = [f for f in os.listdir(augmented_images_dir) if f.endswith('.jpg') or f.endswith('.png')]

    if num_images > len(image_files):
        print(f"Number of requested images ({num_images}) exceeds available images ({len(image_files)}).")
        num_images = len(image_files)

    selected_images = random.sample(image_files, num_images)

    for image_file in selected_images:
        image_path = os.path.join(augmented_images_dir, image_file)
        label_file = os.path.join(augmented_labels_dir, os.path.splitext(image_file)[0] + '.txt')

        if not os.path.exists(label_file):
            print(f"Label file not found for image {image_file}, skipping...")
            continue

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes, labels = load_bboxes(label_file)

        # Apply augmentations and plot
        apply_augmentations_and_plot(image, bboxes, labels)

# Define augmentations
augmentations = A.Compose([
    A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.8),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    A.RandomBrightnessContrast(p=0.8),
    A.HueSaturationValue(p=0.8),
    A.Blur(blur_limit=3, p=0.5),
    A.ToGray(p=1),
    A.RandomScale(scale_limit=0.2, p=0.8),
], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))

# User input
try:
    num_images = int(input("Enter the number of images to randomly select and plot: "))
except ValueError:
    print("Invalid input. Please enter a valid integer.")
    num_images = 0

# Directory paths
augmented_images_dir = 'augmented_images7'
augmented_labels_dir = 'augmented_labels7'

if num_images > 0:
    pick_and_plot_random_images(augmented_images_dir, augmented_labels_dir, num_images)

