import os, shutil

src = "/home/sp2583/rlvr"
dst = "/home/sp2583/rlvr_a6000"

STEP = 50
MAX_STEP = 500      # set to None to allow any multiple of 50
MAX_BYTES = 1 << 30 

def is_global_step_dir(name: str) -> bool:
    if not name.startswith("global_step_"):
        return False
    s = name.split("_", 2)[-1]
    if not s.isdigit():
        return False
    n = int(s)
    if n % STEP != 0:
        return False
    return (MAX_STEP is None) or (0 <= n <= MAX_STEP)

def ignore_filter(cur_dir, names):
    ignored = []
    for n in names:
        p = os.path.join(cur_dir, n)
        if os.path.isdir(p) and is_global_step_dir(n):
            ignored.append(n)
        elif os.path.isfile(p):
            try:
                if os.path.getsize(p) > MAX_BYTES:
                    ignored.append(n)
            except OSError:
                # If size can't be read, skip it to be safe
                ignored.append(n)
    return ignored

shutil.copytree(src, dst, ignore=ignore_filter, dirs_exist_ok=True)
