
# Load Images & Generate Pseudo Masks
def load_images_and_masks(image_folder, target_size=(256, 256)):
    images, masks = [], []
    for file in os.listdir(image_folder):
        img_path = os.path.join(image_folder, file)

        # Read image and check if it is valid
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping unreadable file: {img_path}")  # Debugging message
            continue

        img = cv2.resize(img, target_size)
        img = img / 255.0  # Normalize
        images.append(img)

        # Create a pseudo mask (simple thresholding)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  # Adjust threshold if needed
        masks.append(mask.reshape(target_size + (1,)))

    return np.array(images), np.array(masks)

# Load dataset
image_folder = train_dir  # Update path if needed
X, Y = load_images_and_masks(image_folder)
print(f"Loaded {len(X)} images and {len(Y)} masks")
if len(X) == 0:
   print("No need of MRI segmentation")
