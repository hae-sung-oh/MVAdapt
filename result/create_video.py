import subprocess
from glob import glob
from os.path import join, isdir, isfile, abspath
from os import listdir, remove, cpu_count
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor

def create_video(root, args):
    frame_rate = 20
    output_video = join(root, "output.mp4")

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
    
    task_paths = []
    splits = [s for s in listdir(args.root) if isdir(join(args.root, s))]
    
    for split in splits:
        vehicles = [v for v in listdir(join(args.root, split)) if isdir(join(args.root, split, v))]
        
        for vehicle in vehicles:
            try:
                debug_folder_name = f"debug_{vehicle.split('V')[1]}"
                debug_path = join(args.root, split, vehicle, debug_folder_name)
            except IndexError:
                continue

            if not isdir(debug_path):
                continue
            
            routes = [r for r in listdir(debug_path) if isdir(join(debug_path, r))]
            
            for route in routes:
                route_path = join(debug_path, route)
                print(f"Creating video for {route_path}")
                task_paths.append(route_path)

    if not task_paths:
        exit()

    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(lambda path: create_video(path, args), task_paths)