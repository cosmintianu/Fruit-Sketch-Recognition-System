#
## To run pyhtong script to convert from npy to img:
```sh
# 1. First, install python3-full (required for proper venv)
sudo apt install python3-full

# 2. Delete the broken virtual environment
rm -rf .venv

# 3. Recreate the virtual environment
python3 -m venv .venv

# 4. Activate it
source .venv/bin/activate

# 5. Now pip should work
pip install numpy opencv-python

# 6. Run your script
python convert_npy_to_images.py
```

## Set cpp correct version in VS
Right-click your project in Solution Explorer → Properties​

Set Configuration to "All Configurations" (dropdown at top-left)​

Navigate to: Configuration Properties → C/C++ → Language​

Find C++ Language Standard dropdown​

Select ISO C++17 Standard (/std:c++17) or ISO C++ Latest (/std:c++latest)​

Click OK or Apply​

Rebuild your project