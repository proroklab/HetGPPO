#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import os


def compress_videos():
    for subdir, dirs, files in os.walk(""):
        for file in files:
            # print os.path.join(subdir, file)
            filepath = subdir + os.sep + file

            if filepath.endswith(".mp4"):
                os.system(
                    f"ffmpeg -i {filepath} -vcodec libx265 -crf 28 {str(filepath).removesuffix('.mp4')}_compressed.mp4 "
                )


def mp4_to_gif():
    for subdir, dirs, files in os.walk("/Users/Matteo/Downloads"):
        for file in files:
            # print os.path.join(subdir, file)
            filepath = subdir + os.sep + file

            if filepath.endswith(".mp4"):
                os.system(
                    f'ffmpeg -i {filepath} -vf "fps=30" -c:v pam -f image2pipe - | convert -delay 3 - -loop 0 -layers optimize {str(filepath).removesuffix(".mp4")}.gif'
                )


if __name__ == "__main__":
    mp4_to_gif()
