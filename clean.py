import os
import shutil
for root, dirs, files in os.walk('./handframes/Demo_frames'):
    for f in files:
        os.unlink(os.path.join(root, f))
    for d in dirs:
        shutil.rmtree(os.path.join(root, d))

for root, dirs, files in os.walk('./posenet_nodejs_setup-master/Posenet_Frames'):
    for f in files:
        os.unlink(os.path.join(root, f))
    for d in dirs:
        shutil.rmtree(os.path.join(root, d))
        
 for root, dirs, files in os.walk('./keypoints_Posenet'):
    for f in files:
        os.unlink(os.path.join(root, f))
    for d in dirs:
        shutil.rmtree(os.path.join(root, d))
