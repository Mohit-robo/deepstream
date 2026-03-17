import os
import sys
import argparse

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from lib.test.evaluation.tracker import Tracker

def main():
    parser = argparse.ArgumentParser(description='Run tracker on a video file.')
    parser.add_argument('video_path', type=str, help='Path to the video file.')
    parser.add_argument('--tracker_name', type=str, default='sutrack', help='Name of tracking method.')
    parser.add_argument('--tracker_param', type=str, default='sutrack_t224', help='Name of config file.')
    parser.add_argument('--dataset_name', type=str, default='lasot', help='Name of dataset (e.g. lasot).')
    parser.add_argument('--optional_box', type=float, nargs='+', default=None, help='optional_box x y w h.')
    parser.add_argument('--save_results', action='store_true', help='Save tracking results.')

    args = parser.parse_args()

    # Convert optional_box to list of floats if provided
    optional_box = None
    if args.optional_box is not None:
        assert len(args.optional_box) == 4, "optional_box must have 4 coordinates [x, y, w, h]"
        optional_box = [float(v) for v in args.optional_box]

    print(f"Loading tracker {args.tracker_name} with params {args.tracker_param}...")
    tracker = Tracker(args.tracker_name, args.tracker_param, args.dataset_name)
    
    print(f"Running tracker on {args.video_path}...")
    tracker.run_video(videofilepath=args.video_path, optional_box=optional_box, save_results=args.save_results)

if __name__ == '__main__':
    main()
