# import carla
# import cv2
# import numpy as np
# from team_code.vehicle_config import VehicleConfig

# # Initialize the VehicleConfig object
# vehicle_config = VehicleConfig()
# global image_array
# image_array = None  # Initialize the global variable

# global camera_bp
# camera_bp = None

# def process_image(image):
#     """Processes the image from the sensor."""
#     global image_array
#     image_array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))

# def update_camera(camera, command, ego_vehicle):
#     """Updates the camera position based on the command provided, relative to the vehicle."""
#     relative_transform = camera.get_transform()
#     delta = 0.1
#     commands = {
#         "q": carla.Location(0, 0, delta),
#         "e": carla.Location(0, 0, -delta),
#         "a": carla.Location(-delta, 0, 0),
#         "d": carla.Location(delta, 0, 0),
#         "w": carla.Location(0, delta, 0),
#         "s": carla.Location(0, -delta, 0),
#     }
    
#     if command in commands:
#         new_location = relative_transform.location + commands[command]
#         relative_transform.location = new_location
#         camera.set_transform(relative_transform)
#         print(f"Camera updated to relative location: {relative_transform.location}")
#     elif command == "detach":
#         camera.detach()
#         print("Camera detached from the vehicle.")
#     else:
#         print("Invalid command")

# def spawn_camera(world, ego_vehicle, camera_pos):
#     """Spawns a camera attached to the vehicle."""
#     camera = world.spawn_actor(camera_bp, camera_pos, attach_to=ego_vehicle)
#     camera.listen(lambda image: process_image(image))
#     return camera

# def destroy_existing_actors(world, filters):
#     """Destroys existing actors based on the provided filters."""
#     for filter_pattern in filters:
#         for actor in world.get_actors().filter(filter_pattern):
#             actor.destroy()

# def main():
#     global image_array, camera_bp  # Ensure this variable is accessible in the function
#     client = carla.Client("127.0.0.1", 20000)
#     client.set_timeout(10.0)
#     world = client.get_world()
#     blueprint_library = world.get_blueprint_library()
#     camera_bp = blueprint_library.find("sensor.camera.rgb")
#     camera_bp.set_attribute("image_size_x", "1024")
#     camera_bp.set_attribute("image_size_y", "256")
#     camera_bp.set_attribute("fov", "110")

#     # Clean up existing vehicles and cameras
#     destroy_existing_actors(world, ["vehicle.*", "sensor.camera.*"])

#     # Process each vehicle in the configuration list
#     for i, item in enumerate(vehicle_config.config_list):
#         vehicle_bp = blueprint_library.find(item["vehicle_name"])
#         spawn_point = world.get_map().get_spawn_points()[i]
#         ego_vehicle = world.spawn_actor(vehicle_bp, spawn_point)

#         camera_pos = carla.Transform(carla.Location(*item["camera_pos"]))
#         camera = spawn_camera(world, ego_vehicle, camera_pos)

#         print(f"Spawned {item['vehicle_name']} at index {i}")
#         print(f"Initial camera location: {camera.get_location()}")

#         while True:
#             if image_array is not None:
#                 cv2.imshow("image", image_array)
#                 if cv2.waitKey(1) & 0xFF == ord('n'):
#                     break

#             command = input("Enter camera command (q, e, a, d, w, s, r, detach, n, p): ").strip().lower()

#             if command in ["q", "e", "a", "d", "w", "s"]:
#                 update_camera(camera, command, ego_vehicle)
#             elif command == "r":
#                 camera.set_transform(carla.Transform(carla.Location(*item["camera_pos"])))
#                 print("Camera reset to initial position.")
#             elif command == "detach":
#                 update_camera(camera, "detach", ego_vehicle)
#             elif command == "n":
#                 break
#             elif command == "p":
#                 ego_vehicle.destroy()
#                 camera.destroy()
#                 cv2.destroyAllWindows()
#                 return
#             else:
#                 print("Invalid command. Try again.")

#         ego_vehicle.destroy()
#         camera.destroy()

#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

import carla
import cv2
import numpy as np
from team_code.vehicle_config import VehicleConfig

# Initialize the VehicleConfig object
vehicle_config = VehicleConfig()
global image_array
image_array = None 

def process_image(image):
    global image_array
    image_array = np.frombuffer(image.raw_data, dtype=np.uint8)
    image_array = image_array.reshape((image.height, image.width, 4))
    
def destroy_all(world):
    for actor in world.get_actors().filter("vehicle.*"):
        actor.destroy()
    for actor in world.get_actors().filter("sensor.camera.*"):
        actor.stop()
        actor.destroy()
        
def main():
    global image_array
    # Connect to the CARLA server
    client = carla.Client("127.0.0.1", 20000)
    client.set_timeout(10.0)
    world = client.get_world()

    # Spawn the ego vehicle
    blueprint_library = world.get_blueprint_library()
    destroy_all(world)

    index = [35]
    try:
        for i in index:
            vehicle_bp = blueprint_library.find(vehicle_config.config_list[i]["vehicle_name"])
            spawn_point = world.get_map().get_spawn_points()[i]
            ego_vehicle = world.spawn_actor(vehicle_bp, spawn_point)
            
            camera_bp = blueprint_library.find("sensor.camera.rgb")
            camera_bp.set_attribute("image_size_x", "1024")
            camera_bp.set_attribute("image_size_y", "256")
            camera_bp.set_attribute("fov", "110")
            x, y, z = vehicle_config.config_list[i]["camera_pos"]
            
            camera = world.spawn_actor(camera_bp, carla.Transform(carla.Location(x=x, y=y, z=z)), attach_to=ego_vehicle)
            camera.listen(lambda image: process_image(image))
            
            print(f"Spawned {vehicle_config.config_list[i]['vehicle_name']}, index {i}")
            while True:
                if image_array is not None:
                    cv2.imshow("image", image_array)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        destroy_all(world)
                        break
    finally:
        destroy_all(world)
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    main()
