"""
Script to download the Kokoro ONNX model files.
"""

import os
import sys
import urllib.request
import shutil

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
    
    # Download with progress reporting
    def report_progress(block_num, block_size, total_size):
        read_so_far = block_num * block_size
        if total_size > 0:
            percent = read_so_far * 100 / total_size
            s = f"\r{percent:.1f}% ({read_so_far} / {total_size} bytes)"
            sys.stdout.write(s)
            sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, destination, reporthook=report_progress)
        print(f"\nDownload complete: {destination}")
        return True
    except Exception as e:
        print(f"\nError downloading {url}: {e}")
        return False

def main():
    """Main function."""
    # Define the model files to download
    model_files = [
        {
            "url": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx",
            "destination": os.path.join("models", "kokoro", "kokoro-v0_19.onnx")
        },
        {
            "url": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json",
            "destination": os.path.join("models", "kokoro", "voices.json")
        }
    ]
    
    # Download each file
    success = True
    for file_info in model_files:
        if not download_file(file_info["url"], file_info["destination"]):
            success = False
    
    if success:
        print("\nAll model files downloaded successfully.")
        print("You can now run the application with full TTS functionality.")
    else:
        print("\nSome model files could not be downloaded.")
        print("The application will still work, but will use a simple tone instead of actual speech synthesis.")

if __name__ == "__main__":
    main()
