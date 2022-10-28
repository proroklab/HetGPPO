import os

for subdir, dirs, files in os.walk(
    "/Users/Matteo/SynologyDrive/Università/Cambridge/PhD/Papers/HetGPPO/videos"
):
    for file in files:
        # print os.path.join(subdir, file)
        filepath = subdir + os.sep + file

        if filepath.endswith(".mp4"):
            os.system(
                f"ffmpeg -i {filepath} -vcodec libx265 -crf 28 {str(filepath).removesuffix('.mp4')}_compressed.mp4 "
            )
