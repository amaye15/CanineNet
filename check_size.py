import os
import math

def get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if not os.path.islink(file_path):  # skip if it's a symbolic link
                total_size += os.path.getsize(file_path)
    return total_size

# Example usage
folder_path = '/workspaces/CanineNet/'  # replace with your folder path
size_in_bytes = get_folder_size(folder_path)

# Convert to a readable format (optional)
def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

readable_size = convert_size(size_in_bytes)
print(f"Folder Size: {readable_size}")
