train_dir = os.path.join(extract_path, "Training")
test_dir = os.path.join(extract_path, "Testing")
IMG_SIZE = 256
def load_images_and_masks(directory):
    images = []
    masks = []

    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if not os.path.isdir(class_path):
            continue

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            # Load image
            img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
            img = img_to_array(img).astype(np.uint8)  # Convert to uint8

            # Generate pseudo-mask using thresholding
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
            mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

            images.append(img)
            masks.append(mask)

    return np.array(images), np.array(masks)
