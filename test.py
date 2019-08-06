import openpose as op
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--videoPath',
        type=str,
        default='/media/ywj/File/C-Code/OpenPose-PyTorch/examples/video.avi',
        help='The path of the video')
    parser.add_argument(
        '--modelPath',
        type=str,
        default='/media/ywj/File/C-Code/OpenPose-PyTorch/openpose/models/body_pose_model.pth',
        help='The path of the model')
    args = parser.parse_args()

    videoPath = args.videoPath
    modelPath = args.modelPath

    op.estimations.video2skeleton2D(videoPath, modelPath)