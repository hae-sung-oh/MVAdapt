import random
from team_code_mvadapt.physics_config import PhysicsConfig
import carla

class VehicleConfig:
    def __init__(self, config=None):
        self.config = config
        # fmt: off
        # 2 wheels: 4, 10, 16, 17, 19, 34, 36
        self.physics_list = PhysicsConfig().physics_list
        self.config_list = [
            {   # 0
            "vehicle_name": "vehicle.lincoln.mkz_2017",
            "vehicle_extent": [2.4508416652679443, 1.0641621351242065, 0.7553732395172119],
            "camera_pos": [-1.5, 0.0, 2.0], 
            "lidar_pos": [0.0, 0.0, 2.5],
            "physics": self.physics_list[0]
            },
            {   # 1
            "vehicle_name": "vehicle.audi.a2",
            "vehicle_extent": [1.852684736251831, 0.8943392634391785, 0.7745251059532166],
            "camera_pos": [-1.5, 0.0, 2.0],
            "lidar_pos": [0.0, 0.0, 2.5],
            "physics": self.physics_list[1]
            },
            {   # 2
            "vehicle_name": "vehicle.audi.etron",
            "vehicle_extent": [2.427854299545288, 1.0163782835006714, 0.8246796727180481],
            "camera_pos": [-1.5, 0.0, 2.0],
            "lidar_pos": [0.0, 0.0, 2.5],
            "physics": self.physics_list[2]
            },
            {   # 3
            "vehicle_name": "vehicle.audi.tt",
            "vehicle_extent": [2.0906050205230713, 0.9970585703849792, 0.6926480531692505],
            "camera_pos": [-1.5, 0.0, 2.0],
            "lidar_pos": [0.0, 0.0, 2.5],
            "physics": self.physics_list[3]
            },
            {   # 4 # 2 wheels
            "vehicle_name": "vehicle.bh.crossbike",
            "vehicle_extent": [0.7436444163322449, 0.42962872982025146, 0.6382190585136414],
            "camera_pos": [0.0, 0.0, 2.0],
            "lidar_pos": [0.0, 0.0, 2.5],
            "physics": self.physics_list[4]
            },
            {   # 5
            "vehicle_name": "vehicle.bmw.grandtourer",
            "vehicle_extent": [2.3055028915405273, 1.1208566427230835, 0.8336379528045654],
            "camera_pos": [-1.0, 0.0, 2.0],
            "lidar_pos": [0.0, 0.0, 2.5],
            "physics": self.physics_list[5]
            },
            {   # 6
            "vehicle_name": "vehicle.carlamotors.carlacola",
            "vehicle_extent": [2.601919174194336, 1.3134948015213013, 1.2337223291397095],
            "camera_pos": [0.0, 0.0, 2.0],
            # "lidar_pos": [0.0, 0.0, 2.7],
            "lidar_pos": [-2.0, 0.0, 3.0],
            "physics": self.physics_list[6]
            },
            {   # 7 # modified lidar
            "vehicle_name": "vehicle.carlamotors.firetruck",
            "vehicle_extent": [4.234020709991455, 1.4455441236495972, 1.9137061834335327],
            "camera_pos": [4.0, 0.0, 2.0],
            # "lidar_pos": [4.0, 0.0, 3.5],
            "lidar_pos": [0.0, 0.0, 8.0],
            'lidar_lower_fov': -90.0,
            "lidar_upper_fov": 50.0,
            "physics": self.physics_list[7]
            },
            {   # 8
            "vehicle_name": "vehicle.chevrolet.impala",
            "vehicle_extent": [2.6787397861480713, 1.0166014432907104, 0.7053293585777283],
            "camera_pos": [-1.0, 0.0, 2.0],
            "lidar_pos": [0.0, 0.0, 2.5],
            "physics": self.physics_list[8]
            },
            {   # 9
            "vehicle_name": "vehicle.citroen.c3",
            "vehicle_extent": [1.9938424825668335, 0.9254241585731506, 0.8085547685623169],
            "camera_pos": [-1.0, 0.0, 2.0],
            "lidar_pos": [0.0, 0.0, 2.5],
            "physics": self.physics_list[9]
            },
            {   # 10 # 2 wheels
            "vehicle_name": "vehicle.diamondback.century",
            "vehicle_extent": [0.8214218020439148, 0.18625812232494354, 0.7479714751243591],
            "camera_pos": [0.0, 0.0, 2.0],
            "lidar_pos": [0.0, 0.0, 2.5],
            "physics": self.physics_list[10]
            },
            {   # 11
            "vehicle_name": "vehicle.dodge.charger_2020",
            "vehicle_extent": [2.5030298233032227, 1.0485419034957886, 0.7673624753952026],
            "camera_pos": [-1.5, 0.0, 2.0],
            "lidar_pos": [0.0, 0.0, 2.5],
            "physics": self.physics_list[11]
            },
            {   # 12
            "vehicle_name": "vehicle.dodge.charger_police",
            "vehicle_extent": [2.487122058868408, 1.0192005634307861, 0.7710590958595276],
            "camera_pos": [-1.0, 0.0, 2.0],
            "lidar_pos": [0.0, 0.0, 2.5],
            "physics": self.physics_list[12]
            },
            {   # 13
            "vehicle_name": "vehicle.dodge.charger_police_2020",
            "vehicle_extent": [2.6187572479248047, 1.0485419034957886, 0.819191575050354],
            "camera_pos": [0.0, 0.0, 2.0],
            "lidar_pos": [0.0, 0.0, 2.5],
            "physics": self.physics_list[13]
            },
            {   # 14 # modified lidar
            "vehicle_name": "vehicle.ford.ambulance",
            "vehicle_extent": [3.18282151222229, 1.1755871772766113, 1.215687870979309],
            "camera_pos": [3.5, 0.0, 2.0],
            "lidar_pos": [0.0, 0.0, 3.0],
            "lower_fov": -45.0,
            "physics": self.physics_list[14]
            },
            {   # 15
            "vehicle_name": "vehicle.ford.mustang",
            "vehicle_extent": [2.358762502670288, 0.947413444519043, 0.650469958782196],
            "camera_pos": [-1.0, 0.0, 2.0],
            "lidar_pos": [0.0, 0.0, 2.5],
            "physics": self.physics_list[15]
            },
            {   # 16 # 2 wheels
            "vehicle_name": "vehicle.gazelle.omafiets",
            "vehicle_extent": [0.9177202582359314, 0.0, 0.5856836438179016],
            "camera_pos": [0.0, 0.0, 2.0],
            "lidar_pos": [0.0, 0.0, 2.5],
            "physics": self.physics_list[16]
            },
            {   # 17 # 2 wheels
            "vehicle_name": "vehicle.harley-davidson.low_rider",
            "vehicle_extent": [1.1778701543807983, 0.38183942437171936, 0.6382853388786316],
            "camera_pos": [0.0, 0.0, 2.0],
            "lidar_pos": [0.0, 0.0, 2.5],
            "physics": self.physics_list[17]
            },
            {   # 18
            "vehicle_name": "vehicle.jeep.wrangler_rubicon",
            "vehicle_extent": [1.9331103563308716, 0.9525982737541199, 0.9389679431915283],
            "camera_pos": [0.0, 0.0, 2.0],
            "lidar_pos": [0.0, 0.0, 2.5],
            "physics": self.physics_list[18]
            },
            {   # 19 # 2 wheels
            "vehicle_name": "vehicle.kawasaki.ninja",
            "vehicle_extent": [1.0166761875152588, 0.4012899398803711, 0.5996325016021729],
            "camera_pos": [-1.5, 0.0, 2.0],
            "lidar_pos": [0.0, 0.0, 2.5],
            "physics": self.physics_list[19]
            },
            {   # 20
            "vehicle_name": "vehicle.lincoln.mkz_2020",
            "vehicle_extent": [2.44619083404541, 1.115301489830017, 0.7400735020637512],
            "camera_pos": [-1.0, 0.0, 2.0],
            "lidar_pos": [0.0, 0.0, 2.5],
            "physics": self.physics_list[20]
            },
            {   # 21
            "vehicle_name": "vehicle.mercedes.coupe",
            "vehicle_extent": [2.5133883953094482, 1.0757731199264526, 0.8253258466720581],
            "camera_pos": [-1.0, 0.0, 2.0],
            "lidar_pos": [0.0, 0.0, 2.5],
            "physics": self.physics_list[21]
            },
            {   # 22
            "vehicle_name": "vehicle.mercedes.coupe_2020",
            "vehicle_extent": [2.3368194103240967, 1.0011461973190308, 0.7209736704826355],
            "camera_pos": [-1.5, 0.0, 2.0],
            "lidar_pos": [0.0, 0.0, 2.5],
            "physics": self.physics_list[22]
            },
            {   # 23 # modified lidar
            "vehicle_name": "vehicle.mercedes.sprinter",
            "vehicle_extent": [2.957595109939575, 0.9942164421081543, 1.2803276777267456],
            "camera_pos": [1.5, 0.0, 2.0],
            "lidar_pos": [0.0, 0.0, 3.0],
            "physics": self.physics_list[23]
            },
            {   # 24
            "vehicle_name": "vehicle.micro.microlino",
            "vehicle_extent": [1.1036475896835327, 0.7404598593711853, 0.6880123615264893],
            "camera_pos": [-1.5, 0.0, 2.0],
            "lidar_pos": [0.0, 0.0, 2.5],
            "physics": self.physics_list[24]
            },
            {   # 25 
            "vehicle_name": "vehicle.mini.cooper_s",
            "vehicle_extent": [1.9029000997543335, 0.985137939453125, 0.7375151515007019],
            "camera_pos": [-1.5, 0.0, 2.0],
            "lidar_pos": [0.0, 0.0, 2.5],
            "physics": self.physics_list[25]
            },
            {   # 26 # modified camera
            "vehicle_name": "vehicle.mini.cooper_s_2021",
            "vehicle_extent": [2.2763495445251465, 1.0485360622406006, 0.8835831880569458],
            "camera_pos": [0.0, 0.0, 2.0],
            "lidar_pos": [0.0, 0.0, 2.5],
            "physics": self.physics_list[26]
            },
            {   # 27
            "vehicle_name": "vehicle.nissan.micra",
            "vehicle_extent": [1.8166879415512085, 0.9225568771362305, 0.7506412863731384],
            "camera_pos": [-1.5, 0.0, 2.0],
            "lidar_pos": [0.0, 0.0, 2.5],
            "physics": self.physics_list[27]
            },
            {   # 28 
            "vehicle_name": "vehicle.nissan.patrol",
            "vehicle_extent": [2.3022549152374268, 0.9657964706420898, 0.9274230599403381],
            "camera_pos": [2.0, 0.0, 2.0],
            "lidar_pos": [0.0, 0.0, 2.5],
            "physics": self.physics_list[28]
            },
            {   # 29
            "vehicle_name": "vehicle.nissan.patrol_2021",
            "vehicle_extent": [2.782914400100708, 1.0749834775924683, 1.0225735902786255],
            "camera_pos": [2.0, 0.0, 2.0],
            "lidar_pos": [0.0, 0.0, 2.5],
            "physics": self.physics_list[29]
            },
            {   # 30
            "vehicle_name": "vehicle.seat.leon",
            "vehicle_extent": [2.0964150428771973, 0.9080929160118103, 0.7369155883789062],
            "camera_pos": [-1.5, 0.0, 2.0],
            "lidar_pos": [0.0, 0.0, 2.5],
            "physics": self.physics_list[30]
            },
            {   # 31 # modified lidar
            "vehicle_name": "vehicle.tesla.cybertruck",
            "vehicle_extent": [3.1367764472961426, 1.1947870254516602, 1.049095630645752],
            "camera_pos": [2.0, 0.0, 2.0],
            "lidar_pos": [0.0, 0.0, 3.0],
            "lower_fov": -45.0,
            "physics": self.physics_list[31]
            },
            {   # 32
            "vehicle_name": "vehicle.tesla.model3",
            "vehicle_extent": [2.3958897590637207, 1.081725001335144, 0.744159996509552],
            "camera_pos": [-1.5, 0.0, 2.0],
            "lidar_pos": [0.0, 0.0, 2.5],
            "physics": self.physics_list[32]
            },
            {   # 33
            "vehicle_name": "vehicle.toyota.prius",
            "vehicle_extent": [2.256761312484741, 1.0034072399139404, 0.7624167203903198],
            "camera_pos": [-1.0, 0.0, 2.0],
            "lidar_pos": [0.0, 0.0, 2.5],
            "physics": self.physics_list[33]
            },
            {   # 34 # 2 wheels
            "vehicle_name": "vehicle.vespa.zx125",
            "vehicle_extent": [0.9023334980010986, 0.0, 0.6178141832351685],
            "camera_pos": [0.0, 0.0, 2.0],
            "lidar_pos": [0.0, 0.0, 2.5],
            "physics": self.physics_list[34]
            },
            {   # 35 # modified lidar
            "vehicle_name": "vehicle.volkswagen.t2",
            "vehicle_extent": [2.2402184009552, 1.034657597541809, 1.0188959836959839],
            "camera_pos": [1.5, 0.0, 2.0],
            "lidar_pos": [1.5, 0.0, 2.5],
            "physics": self.physics_list[35]
            },
            {   # 36 # 2 wheels
            "vehicle_name": "vehicle.yamaha.yzf",
            "vehicle_extent": [1.1047229766845703, 0.43351709842681885, 0.6255727410316467],
            "camera_pos": [0.0, 0.0, 2.0],
            "lidar_pos": [0.0, 0.0, 2.5],
            "physics": self.physics_list[36]
            }
        ]
    
    def update_config(self, config):
        self.config = config
        
    def sample_random_physics(self, actor, index):
        print("Sample random physics")
        assert actor.type_id == self.config_list[index]["vehicle_name"], "Actor type_id does not match"

        physics = actor.get_physics_control()

        physics.max_rpm = random.uniform(self.config.min_max_rpm, self.config.max_max_rpm)
        physics.mass = random.uniform(self.config.min_mass, self.config.max_mass)  
        physics.center_of_mass = carla.Location(x=random.uniform(-1, 1), y=0.0, z=random.uniform(-1, 1)) 

        max_torque = min(random.randint(1, 4), len(physics.torque_curve))
        physics.torque_curve = [
            carla.Vector2D(
                x=physics.torque_curve[i].x * random.uniform(0.8, 1.2), 
                y=physics.torque_curve[i].y * random.uniform(0.8, 1.2)) for i in range(max_torque)
            ]

        for wheel in physics.wheels:
            wheel.radius = random.uniform(20, 50)

        max_gear = min(random.randint(1, self.config.max_gear_num), len(physics.forward_gears))
        physics.forward_gears = [
            carla.GearPhysicsControl(
                ratio=physics.forward_gears[i].ratio * random.uniform(0.8, 1.2),
                down_ratio=physics.forward_gears[i].down_ratio * random.uniform(0.8, 1.2),
                up_ratio=physics.forward_gears[i].up_ratio * random.uniform(0.8, 1.2)
            ) for i in range(max_gear)
        ]
        
        actor.apply_physics_control(physics)

        self.config_list[index] = {
                "vehicle_name": actor.type_id,
                "vehicle_extent": self.config_list[index]["vehicle_extent"],
                "camera_pos": self.config_list[index]["camera_pos"],
                "physics": self.physics2dict(physics)
            }

    def physics2dict(self, physics):
        return {
            "torque_curve": [(p.x, p.y) for p in physics.torque_curve],
            "max_rpm": physics.max_rpm,
            "moi": physics.moi,
            "damping_rate_full_throttle": physics.damping_rate_full_throttle,
            "damping_rate_zero_throttle_clutch_engaged": physics.damping_rate_zero_throttle_clutch_engaged,
            "damping_rate_zero_throttle_clutch_disengaged": physics.damping_rate_zero_throttle_clutch_disengaged,
            "use_gear_autobox": physics.use_gear_autobox,
            "gear_switch_time": physics.gear_switch_time,
            "clutch_strength": physics.clutch_strength,
            "final_ratio": physics.final_ratio,
            "forward_gears": [
                {"ratio": g.ratio, "down_ratio": g.down_ratio, "up_ratio": g.up_ratio} for g in physics.forward_gears
            ],
            "mass": physics.mass,
            "drag_coefficient": physics.drag_coefficient,
            "center_of_mass": (physics.center_of_mass.x, physics.center_of_mass.y, physics.center_of_mass.z),
            "steering_curve": [(p.x, p.y) for p in physics.steering_curve],
            "wheels": [
                {
                    "tire_friction": w.tire_friction,
                    "damping_rate": w.damping_rate,
                    "max_steer_angle": w.max_steer_angle,
                    "radius": w.radius,
                    "position": (w.position.x, w.position.y, w.position.z),
                    "max_brake_torque": w.max_brake_torque,
                    "max_handbrake_torque": w.max_handbrake_torque,
                } for w in physics.wheels
            ]
        }

    def dict2physics(self, dict):
        physics = carla.VehiclePhysicsControl()
        physics.torque_curve = [carla.Vector2D(x=p[0], y=p[1]) for p in dict["torque_curve"]]
        physics.max_rpm = dict["max_rpm"]
        physics.moi = dict["moi"]
        physics.damping_rate_full_throttle = dict["damping_rate_full_throttle"]
        physics.damping_rate_zero_throttle_clutch_engaged = dict["damping_rate_zero_throttle_clutch_engaged"]
        physics.damping_rate_zero_throttle_clutch_disengaged = dict["damping_rate_zero_throttle_clutch_disengaged"]
        physics.use_gear_autobox = dict["use_gear_autobox"]
        physics.gear_switch_time = dict["gear_switch_time"]
        physics.clutch_strength = dict["clutch_strength"]
        physics.final_ratio = dict["final_ratio"]
        physics.forward_gears = [
            carla.GearPhysicsControl(ratio=g["ratio"], down_ratio=g["down_ratio"], up_ratio=g["up_ratio"]) for g in dict["forward_gears"]
        ]
        physics.mass = dict["mass"]
        physics.drag_coefficient = dict["drag_coefficient"]
        physics.center_of_mass = carla.Location(dict["center_of_mass"][0], dict["center_of_mass"][1], dict["center_of_mass"][2])
        physics.steering_curve = [carla.Vector2D(x=p[0], y=p[1]) for p in dict["steering_curve"]]
        physics.wheels = [
            carla.WheelPhysicsControl(
                tire_friction=w["tire_friction"],
                damping_rate=w["damping_rate"],
                max_steer_angle=w["max_steer_angle"],
                radius=w["radius"],
                position=carla.Vector3D(w["position"][0], w["position"][1], w["position"][2]),
                max_brake_torque=w["max_brake_torque"],
                max_handbrake_torque=w["max_handbrake_torque"]
            ) for w in dict["wheels"]
        ]
        return physics

