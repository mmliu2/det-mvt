import os

# Replace this with your target directory
root_dir = '/mnt/data/depthtrack/train'

for dirpath, dirnames, filenames in os.walk(root_dir):
    for dirname in dirnames:
        if 'class_feats' in dirname:
            dir_path = os.path.join(dirpath, dirname)
            try:
                os.rmdir(dir_path)
                print(f"Deleted: {dir_path}")
            except Exception as e:
                print(f"Failed to delete {dir_path}: {e}")
    for filename in filenames:
        if '.npz' in filename:
            file_path = os.path.join(dirpath, filename)
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
