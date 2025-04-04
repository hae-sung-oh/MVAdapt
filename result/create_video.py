import subprocess
from glob import glob
from os.path import join, isdir, isfile
from os import listdir, remove
from argparse import ArgumentParser

def create_video(root, args):
    frame_rate = 20
    output_video = join(root, "output.mp4")

    # Ensure images exist
    image_files = sorted(glob(join(root, "*.png")))
    if not image_files:
        print(f"No PNG images found in {root}, skipping...")
        return
    
    if args.no_overwrite and isfile(output_video):
        option = "-n"
    else:
        option = "-y"

    # Construct FFmpeg command
    ffmpeg_cmd = [
        "ffmpeg",
        option,  # Overwrite output file if it exists
        "-framerate", str(frame_rate),  # Set frame rate
        "-pattern_type", "glob",  # Use wildcard pattern
        "-i", join(root, "*.png"),  # Input pattern
        "-c:v", "libx264",  # Encode in H.264
        "-pix_fmt", "yuv420p",  # Set pixel format
        "-loglevel", "fatal",  # Suppress output   
        output_video  # Output video path
    ]

    # Run FFmpeg command
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"Video created at {output_video}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating video in {root}: {e}")
        
    if args.clear:
        clear_folder(root)
        
def clear_folder(root):
    for file in listdir(root):
        if file.endswith(".png"):
            remove(join(root, file))
    print(f"Removed all PNG images in {root}")

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("root", type=str)
    args.add_argument("--clear", action="store_true")
    args.add_argument("--no-overwrite", action="store_false")
    args = args.parse_args()
    
    splits = [s for s in listdir(args.root) if isdir(join(args.root, s))]
    
    for split in splits:
        vehicles = [v for v in listdir(join(args.root, split)) if isdir(join(args.root, split, v))]
        
        for vehicle in vehicles:
            debug = join(args.root, split, vehicle, f"debug_{vehicle[1]}")
            if not isdir(debug):
                continue
            
            routes = [r for r in listdir(debug) if isdir(join(debug, r))]
            
            for route in routes:
                route_path = join(debug, route)
                print(f"Creating video for {route_path}")
                create_video(route_path, args)
