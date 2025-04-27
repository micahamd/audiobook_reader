"""
Script to copy Kokoro ONNX model files from an existing project.
"""

import os
import shutil
import sys

def copy_file(source, destination):
    """
    Copy a file from source to destination.
    
    Args:
        source: Source file path.
        destination: Destination file path.
    """
    print(f"Copying {source} to {destination}...")
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    try:
        shutil.copy2(source, destination)
        print(f"Copy complete: {destination}")
        return True
    except Exception as e:
        print(f"Error copying {source}: {e}")
        return False

def main():
    """Main function."""
    # Define the source directory
    source_dir = os.path.expanduser("C:/Users/micah/Downloads/Python Proj/kokoro-onnx/new_project")
    
    # Define the model files to copy
    model_files = [
        {
            "source": os.path.join(source_dir, "kokoro-v0_19.onnx"),
            "destination": os.path.join("models", "kokoro", "kokoro-v0_19.onnx")
        },
        {
            "source": os.path.join(source_dir, "voices.json"),
            "destination": os.path.join("models", "kokoro", "voices.json")
        }
    ]
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"Error: Source directory {source_dir} does not exist.")
        return
    
    # Copy each file
    success = True
    for file_info in model_files:
        if not os.path.exists(file_info["source"]):
            print(f"Error: Source file {file_info['source']} does not exist.")
            success = False
            continue
            
        if not copy_file(file_info["source"], file_info["destination"]):
            success = False
    
    if success:
        print("\nAll model files copied successfully.")
        print("You can now run the application with full TTS functionality.")
    else:
        print("\nSome model files could not be copied.")
        print("The application will still work, but will use a simple tone instead of actual speech synthesis.")
        print("You can try downloading the model files instead by running: python download_models.py")

if __name__ == "__main__":
    main()
