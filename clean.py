import os
import shutil
for root, dirs, files in os.walk('./handframes/Demo_frames'):
    for f in files:
        os.unlink(os.path.join(root, f))
    for d in dirs:
        shutil.rmtree(os.path.join(root, d))