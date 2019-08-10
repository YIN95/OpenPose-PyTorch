import openpose as op
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--videoPath',
        type=str,
        default='/media/ywj/File/C-Code/OpenPose-PyTorch/examples/video',
        help='The path of the video')
    parser.add_argument(
        '--saveImagePath',
        type=str,
        default='/media/ywj/File/C-Code/OpenPose-PyTorch/examples/skeletonImages',
        help='The path of the skeleton images')
    parser.add_argument(
        '--modelPath',
        type=str,
        default='/media/ywj/File/C-Code/OpenPose-PyTorch/openpose/models/body_pose_model.pth',
        help='The path of the model')
    args = parser.parse_args()

    videoPath = args.videoPath
    modelPath = args.modelPath
    saveImagePath = args.saveImagePath

    op.estimations.video2skeleton2D(videoPath, modelPath, saveImages=True, saveImages_path=saveImagePath)