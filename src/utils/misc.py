import re
import glob

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def get_files(glob_pattern, base_path, fil):
    files = glob.glob(glob_pattern, root_dir=base_path)
    files = filter(lambda x: fil in x, files)
    return list(natural_sort(files))
