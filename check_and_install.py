import json
import subprocess
import sys

# Function to check if a library is installed
def is_library_installed(library_name):
    try:
        # Try to import the library to see if it exists
        __import__(library_name)
        return True
    except ImportError:
        return False

# Optionally, check if these libraries exist in the current Python environment
def check_library_in_environment(library_name):
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "show", library_name], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"'{library_name}' is installed in the current environment:")
            print(result.stdout)
        else:
            print(f"'{library_name}' is not installed in the current environment.")
            sys.exit(1)
    except Exception as e:
        print(f"An error occurred while checking the environment for '{library_name}': {e}")
        sys.exit(1)

# Load the list of required packages from requirements.json
with open("requirements.json", "r") as f:
    data = json.load(f)
    required_packages = data.get("libraries", [])

# Check and install missing packages
for package in required_packages:
    # Check if the package is already installed
    if not is_library_installed(package.lower()):
        print(f"Installing '{package}'...")
        try:
            # Install the package if not found
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}: {e}")
            sys.exit(1)
    else:
        print(f"'{package}' is already installed.")

# Check libraries in the environment after installation
for package in required_packages:
    check_library_in_environment(package.lower())