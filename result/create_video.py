import subprocess
from glob import glob
from os.path import join, isdir, isfile, abspath
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
    
    if args.overwrite and isfile(output_video):
        option = "-n"
    else:
        option = "-y"

    list_path = join(root, "image_files.txt")
    with open(list_path, "w") as f:
        for file in image_files:
            f.write(f"file '{abspath(file)}'\n")

    ffmpeg_cmd = [
        "ffmpeg",
        option,
        "-f", "concat",
        "-safe", "0",
        "-r", str(frame_rate),
        "-i", list_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-loglevel", "fatal",
        output_video
    ]

    # Run FFmpeg command
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"Video created at {output_video}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating video in {root}: {e}")
        
    if args.clear:
        clear_folder(root)
    if isfile(list_path):
        remove(list_path)
        
def clear_folder(root):
    for file in listdir(root):
        if file.endswith(".png"):
            remove(join(root, file))
    print(f"Removed all PNG images in {root}")

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("root", type=str)
    args.add_argument("--clear", action="store_true")
    args.add_argument("--overwrite", action="store_true")
    args = args.parse_args()
    
    splits = [s for s in listdir(args.root) if isdir(join(args.root, s))]
    
    for split in splits:
        vehicles = [v for v in listdir(join(args.root, split)) if isdir(join(args.root, split, v))]
        
        for vehicle in vehicles:
            debug = join(args.root, split, vehicle, f"debug_{vehicle.split('V')[1]}")
            if not isdir(debug):
                continue
            
            routes = [r for r in listdir(debug) if isdir(join(debug, r))]
            
            for route in routes:
                route_path = join(debug, route)
                print(f"Creating video for {route_path}")
                create_video(route_path, args)
