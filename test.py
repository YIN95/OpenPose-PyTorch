import openpose as op


if __name__ == "__main__":
    video_path = '/media/ywj/File/C-Code/OpenPose-PyTorch/examples/video.avi'
    op.estimations.video2skeleton2D(video_path)