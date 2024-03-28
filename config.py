import os

# Check for a Kaggle-specific environment variable or path
if os.path.exists('/kaggle/input'):
    # Running on Kaggle
    BASE_PATH = '/kaggle/input/baid-model-test/baid-model'
    SAVE_DIR = '/kaggle/working/result'
else:
    # Assume running locally. Dynamically set BASE_PATH based on the script's location.
    # This sets BASE_PATH to the script's directory, allowing for flexibility across different local environments.
    BASE_PATH = os.path.dirname(os.path.realpath(__file__))
    SAVE_DIR = os.path.join(BASE_PATH, 'result')

# Additional configuration variables can be added here
