# import random
# import cv2
# import os
# import albumentations as A
#
#
# # Function to draw bounding boxes on the image
# def draw_bboxes(image, bboxes, labels):
#     h, w, _ = image.shape
#     for bbox, label in zip(bboxes, labels):
#         # YOLO format (x_center, y_center, width, height) -> (x_min, y_min, x_max, y_max)
#         x_center, y_center, width, height = bbox
#         x_min = int((x_center - width / 2) * w)
#         y_min = int((y_center - height / 2) * h)
#         x_max = int((x_center + width / 2) * w)
#         y_max = int((y_center + height / 2) * h)
#
#         # Draw rectangle and label on the image
#         cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#         cv2.putText(image, f'Class {label}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#     return image
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
# # Function to randomly pick and display images with augmentations applied
# def pick_and_augment_images(augmented_images_dir, augmented_labels_dir, augmentations, bbox_params, num_images):
#     # Get list of all images in augmented directory
#     augmented_images = [f for f in os.listdir(augmented_images_dir) if f.endswith(('.jpg', '.png'))]
#
#     # Randomly select the user-specified number of images
#     selected_images = random.sample(augmented_images, min(num_images, len(augmented_images)))
#
#     for image_file in selected_images:
#         # Load image
#         image_path = os.path.join(augmented_images_dir, image_file)
#         image = cv2.imread(image_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
#
#         # Load corresponding label file
#         label_path = os.path.join(augmented_labels_dir, os.path.splitext(image_file)[0] + '.txt')
#         if not os.path.exists(label_path):
#             print(f"Label file not found for image {image_file}, skipping...")
#             continue
#
#         # Load bounding boxes and labels
#         bboxes, labels = load_bboxes(label_path)
#
#         # Apply the augmentation (use albumentations for rotations and bounding box transformations)
#         augmented = A.Compose(augmentations, bbox_params=bbox_params)(image=image, bboxes=bboxes, category_ids=labels)
#
#         # Get the augmented image and transformed bounding boxes
#         augmented_image = augmented['image']
#         augmented_bboxes = augmented['bboxes']
#         augmented_labels = augmented['category_ids']
#
#         # Draw the transformed bounding boxes on the augmented image
#         image_with_bboxes = draw_bboxes(augmented_image, augmented_bboxes, augmented_labels)
#
#         # Display the augmented image
#         cv2.imshow(f"Augmented Image: {image_file}", cv2.cvtColor(image_with_bboxes, cv2.COLOR_RGB2BGR))
#         cv2.waitKey(0)  # Press any key to close the window
#         cv2.destroyAllWindows()
#
#
# # User input for number of random images to select
# num_images = int(input("Enter the number of random augmented images to pick: "))
#
# # Define the augmentations (ensure you include the transformations that handle rotations)
# augmentations = [
#     A.Rotate(limit=40, p=1),  # Example: 40-degree random rotation
#     # Add more augmentations here if needed
# ]
#
# # Bounding box parameters in YOLO format (adjust as per your use case)
# bbox_params = A.BboxParams(format='yolo', label_fields=['category_ids'], min_visibility=0.1)
# augmented_images_dir='augmented_images7'
# augmented_labels_dir='augmented_labels7'
# # Call the function to pick, augment, and display images with bounding boxes
# pick_and_augment_images(augmented_images_dir, augmented_labels_dir, augmentations, bbox_params, num_images)

import cv2
import random
import os
import matplotlib.pyplot as plt
import numpy as np

# Function to rotate a point around a center by a given angle
def rotate_point(cx, cy, angle, px, py):
    radians = np.deg2rad(angle)
    cos = np.cos(radians)
    sin = np.sin(radians)

    # Translate point to origin
    px -= cx
    py -= cy

    # Rotate point
    x_new = px * cos - py * sin
    y_new = px * sin + py * cos

    # Translate point back
    px = x_new + cx
    py = y_new + cy
    return px, py

# Function to rotate the bounding boxes
def rotate_bboxes(bboxes, img_w, img_h, angle):
    cx, cy = img_w / 2, img_h / 2  # Image center
    rotated_bboxes = []

    for bbox in bboxes:
        x_center, y_center, width, height = bbox

        # Calculate the coordinates of the four corners of the bounding box
        x_min = (x_center - width / 2) * img_w
        y_min = (y_center - height / 2) * img_h
        x_max = (x_center + width / 2) * img_w
        y_max = (y_center + height / 2) * img_h

        corners = [
            (x_min, y_min), (x_max, y_min),
            (x_max, y_max), (x_min, y_max)
        ]

        # Rotate each corner of the bounding box
        rotated_corners = [rotate_point(cx, cy, angle, x, y) for x, y in corners]

        # Find the new bounding box from the rotated corners
        xs, ys = zip(*rotated_corners)
        x_min_new, x_max_new = min(xs), max(xs)
        y_min_new, y_max_new = min(ys), max(ys)

        # Convert back to normalized format (center, width, height)
        new_x_center = (x_min_new + x_max_new) / 2 / img_w
        new_y_center = (y_min_new + y_max_new) / 2 / img_h
        new_width = (x_max_new - x_min_new) / img_w
        new_height = (y_max_new - y_min_new) / img_h

        rotated_bboxes.append((new_x_center, new_y_center, new_width, new_height))

    return rotated_bboxes

# Function to plot the image with bounding boxes
def plot_image_with_boxes(image, bboxes, angle):
    plt.figure(figsize=(10, 10))
    img_h, img_w, _ = image.shape

    if angle != 0:
        # Get image center and rotate
        center = (img_w // 2, img_h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (img_w, img_h))
        bboxes = rotate_bboxes(bboxes, img_w, img_h, angle)  # Rotate bounding boxes

    plt.imshow(image)

    for bbox in bboxes:
        x_center, y_center, width, height = bbox
        x_min = int((x_center - width / 2) * img_w)
        y_min = int((y_center - height / 2) * img_h)
        x_max = int((x_center + width / 2) * img_w)
        y_max = int((y_center + height / 2) * img_h)
        plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                          edgecolor='red', facecolor='none', linewidth=2))

    plt.axis('off')  # Hide axis
    plt.show()

# Function to randomly pick images, rotate them, and plot
def pick_and_plot_random_images(augmented_images_dir, augmented_labels_dir, num_images, angle):
    image_files = [f for f in os.listdir(augmented_images_dir) if f.endswith('.jpg') or f.endswith('.png')]

    if num_images > len(image_files):
        print(f"Number of requested images ({num_images}) exceeds available images ({len(image_files)}).")
        num_images = len(image_files)  # Adjust to available images if necessary

    selected_images = random.sample(image_files, num_images)  # Randomly select the specified number of images

    for image_file in selected_images:
        image_path = os.path.join(augmented_images_dir, image_file)
        label_file = os.path.join(augmented_labels_dir, os.path.splitext(image_file)[0] + '.txt')

        if not os.path.exists(label_file):
            print(f"Label file not found for image {image_file}, skipping...")
            continue

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes, _ = load_bboxes(label_file)  # Assuming load_bboxes loads bounding boxes in normalized format

        plot_image_with_boxes(image, bboxes, angle)

# Example of loading bounding boxes (you may need to adjust depending on your format)
def load_bboxes(label_file):
    bboxes = []
    with open(label_file, 'r') as f:
        for line in f:
            # Assuming YOLO format: class_id x_center y_center width height
            _, x_center, y_center, width, height = map(float, line.strip().split())
            bboxes.append((x_center, y_center, width, height))
    return bboxes, None  # None is for any extra data (like class_id) if needed

def save_bboxes(label_file, bboxes, labels):
    with open(label_file, 'w') as file:
        for bbox, label in zip(bboxes, labels):
            x_center, y_center, width, height = bbox
            file.write(f"{label} {x_center} {y_center} {width} {height}\n")

# Directory paths (set according to your setup)
augmented_images_dir = 'augmented_images7'
augmented_labels_dir = 'augmented_labels7'

# Fixed rotation angle (you can change this to any angle you want)
rotation_angle = 20  # Rotate by 45 degrees for example

# Pick and plot 5 random images with rotated bounding boxes
pick_and_plot_random_images(augmented_images_dir, augmented_labels_dir, num_images=5, angle=rotation_angle)