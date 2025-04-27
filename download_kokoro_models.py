"""
Download Kokoro ONNX model files.
"""

import os
import urllib.request
import sys

def download_file(url, destination):
    """
    Download a file from a URL to a destination.
    
    Args:
        url: The URL to download from.
        destination: The destination file path.
    """
    print(f"Downloading {url} to {destination}...")
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Download the file
    try:
        urllib.request.urlretrieve(url, destination)
        print(f"Downloaded {destination}")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {str(e)}")
        return False

def main():
    """Main function."""
    # Define the model files
    model_files = {
        "kokoro-v0_19.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx",
        "voices.json": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json"
    }
    
    # Define the destination directory
    dest_dir = os.path.join(os.path.dirname(__file__), "models", "kokoro")
    
    # Create the directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Download the files
    success = True
    for filename, url in model_files.items():
        dest_path = os.path.join(dest_dir, filename)
        if os.path.exists(dest_path):
            print(f"{dest_path} already exists, skipping download")
        else:
            success = download_file(url, dest_path) and success
    
    if success:
        print("All model files downloaded successfully")
        return 0
    else:
        print("Failed to download some model files")
        return 1

if __name__ == "__main__":
    sys.exit(main())
