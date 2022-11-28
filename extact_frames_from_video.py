import cv2
import os
import os.path as P
from glob import glob
from multiprocessing import Pool
from functools import partial

from params import VideoHyperParams


def extract_frames(video_path, output_dir, width, height):
    save_dir = P.join(output_dir, P.basename(video_path).split('.')[0])
    os.makedirs(save_dir, exist_ok=True)
    video = cv2.VideoCapture(video_path)
    num = 1
    while video.isOpened():
        reg, img = video.read()
        if not reg:
            break
        img = cv2.resize(img, (width, height))
        cv2.imwrite(P.join(save_dir, f"img_{num:05d}.jpg"), img)
        num += 1
    if num < 215:
        print(video_path)


if __name__ == '__main__':
    OUTPUT_DIR = 'features_data/hammer/video_features'
    INPUT_DIR = 'converted_data/hammer/videos_10s_21.5fps'

    video_paths = glob(P.join(INPUT_DIR, "*.mp4"))
    video_paths.sort()
    with Pool(1) as p:
        p.map(partial(extract_frames, output_dir=OUTPUT_DIR,
                    width=VideoHyperParams.FRAME_WIDTH, height=VideoHyperParams.FRAME_HEIGHT), video_paths)

