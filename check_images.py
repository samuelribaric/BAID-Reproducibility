import os
import pandas as pd


# Assuming BASE_PATH is the root directory for the project
BASE_PATH = "/Users/samuelribaric/Desktop/Kurser/TU Delft/Deep Learning/Project/Repo/BAID-Reproducibility"
images_dir = os.path.join(BASE_PATH, "images")
datasets_dir = os.path.join(BASE_PATH, "dataset")

# List of dataset CSV files
datasets = ["train_set.csv", "val_set.csv", "test_set.csv"]

# Load and combine all datasets to get a complete list of expected images
all_images = set()
for dataset in datasets:
    df = pd.read_csv(os.path.join(datasets_dir, dataset))
    all_images.update(df['image'].tolist())

# Get a list of all actual images
actual_images = {os.path.basename(path) for path in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, path))}

# Find missing and unreferenced images
missing_images = all_images - actual_images
unreferenced_images = actual_images - all_images

# Report
if missing_images:
    print("Missing images (listed in datasets but not found):")
    for img in missing_images:
        print(img)
else:
    print("No missing images found.")

if unreferenced_images:
    print("\nUnreferenced images (present in folder but not listed in any dataset):")
    for img in unreferenced_images:
        print(img)
else:
    print("\nNo unreferenced images found.")




# Assuming BASE_PATH is the root directory for the project
BASE_PATH = "/Users/samuelribaric/Desktop/Kurser/TU Delft/Deep Learning/Project/Repo/BAID-Reproducibility"
images_dir = os.path.join(BASE_PATH, "images")
datasets_dir = os.path.join(BASE_PATH, "dataset")

# List of dataset CSV files
datasets = ["train_set.csv", "val_set.csv", "test_set.csv"]

# Set of missing images identified by your verification function
missing_images = {
    "104225.jpg", "108747.jpg", "294478.jpg", "201603.jpg",
    "27862.jpg", "60039.jpg", "199856.jpg", "201446.jpg"
}

# Function to remove missing images from dataset
def remove_missing_images(dataset_path, missing_images):
    # Load the dataset
    df = pd.read_csv(dataset_path)
    
    # Filter out rows corresponding to missing images
    filtered_df = df[~df['image'].isin(missing_images)]
    
    # Save the cleaned dataset back to the file
    filtered_df.to_csv(dataset_path, index=False)

# Iterate over each dataset and remove missing images
for dataset in datasets:
    dataset_path = os.path.join(datasets_dir, dataset)
    remove_missing_images(dataset_path, missing_images)

print("Missing images have been removed from the datasets.")