#!/usr/bin/env python
# Copyright (c) 2018-2019 Intel Corporation.
# authors: German Ros (german.ros@intel.com), Felipe Codevilla (felipe.alcm@gmail.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
CARLA Challenge Evaluator Routes

Provisional code to evaluate Autonomous Agents for the CARLA Autonomous Driving challenge
"""
from __future__ import print_function

import pickle
import trace
import traceback
import argparse
from argparse import RawTextHelpFormatter
from datetime import datetime
import importlib
import os
import cv2
import numpy as np
import sys
import json

import pygame

import carla
import signal

from srunner.scenariomanager.carla_data_provider import *
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from leaderboard.scenarios.scenario_manager_local import ScenarioManager
from leaderboard.scenarios.route_scenario_local import RouteScenario
from leaderboard.envs.sensor_interface import SensorConfigurationInvalid
from leaderboard.autoagents.agent_wrapper_local import AgentWrapper, AgentError
from leaderboard.utils.statistics_manager_local import StatisticsManager
from leaderboard.utils.route_indexer import RouteIndexer
from leaderboard.utils.checkpoint_tools import fetch_dict

from team_code_mvadapt.vehicle_config import VehicleConfig

import pathlib

sensors_to_icons = {
    "sensor.camera.rgb": "carla_camera",
    "sensor.lidar.ray_cast": "carla_lidar",
    "sensor.other.radar": "carla_radar",
    "sensor.other.gnss": "carla_gnss",
    "sensor.other.imu": "carla_imu",
    "sensor.opendrive_map": "carla_opendrive_map",
    "sensor.speedometer": "carla_speedometer",
    "sensor.stitch_camera.rgb": "carla_camera",  # for local World on Rails evaluation
    "sensor.camera.semantic_segmentation": "carla_camera",  # for datagen
    "sensor.camera.depth": "carla_camera",  # for datagen
}
FAILED_LOG = ["Failed - Agent couldn't be set up", "Failed - Agent took too long to setup", "Failed - Agent crashed", "Failed - Simulation crashed"]

class LeaderboardEvaluator(object):
    """
    TODO: document me!
    """

    ego_vehicles = []

    # Tunable parameters
    client_timeout = 1000.0  # in seconds
    wait_for_world = 2000.0  # in seconds
    frame_rate = 20.0  # in Hz

    def __init__(self, args, statistics_manager):
        """
        Setup CARLA client and world
        Setup ScenarioManager
        """
        self.statistics_manager = statistics_manager
        self.sensors = None
        self.sensor_icons = []
        self._vehicle_lights = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam
        self.vehicle_config = VehicleConfig()

        # First of all, we need to create the client that will send the requests
        # to the simulator. Here we'll assume the simulator is accepting
        # requests in the localhost at port 2000.
        self.client = carla.Client(args.host, int(args.port))
        if args.timeout:
            self.client_timeout = float(args.timeout)
        self.client.set_timeout(self.client_timeout)

        try:
            self.world = self.client.load_world("Town01")
        except RuntimeError:
            # For cases where load_world crashes, but the world was properly
            # loaded anyways
            self.world = self.client.get_world()
        self.traffic_manager = self.client.get_trafficmanager(int(args.trafficManagerPort))

        # dist = pkg_resources.get_distribution("carla")
        # if dist.version != "leaderboard":
        #     if LooseVersion(dist.version) < LooseVersion("0.9.10"):
        #         raise ImportError("CARLA version 0.9.10.1 or newer required. CARLA version found: {}".format(dist))

        # Load agent
        module_name = os.path.basename(args.agent).split(".")[0]
        sys.path.insert(0, os.path.dirname(args.agent))
        self.module_agent = importlib.import_module(module_name)

        # Create the ScenarioManager
        self.manager = ScenarioManager(args.timeout, args.debug > 1)

        # Time control for summary purposes
        self._start_time = GameTime.get_time()
        self._end_time = None

        # Create the agent timer
        self._agent_watchdog = Watchdog(int(float(args.timeout)))
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        if self._agent_watchdog and not self._agent_watchdog.get_status():
            raise RuntimeError("Timeout: Agent took too long to setup")
        elif self.manager:
            self.manager.signal_handler(signum, frame)

    def __del__(self):
        """
        Cleanup and delete actors, ScenarioManager and CARLA world
        """

        self._cleanup()
        if hasattr(self, "manager") and self.manager:
            del self.manager
        if hasattr(self, "world") and self.world:
            del self.world

    def _cleanup(self, results=None):
        """
        Remove and destroy all actors
        """

        # Simulation still running and in synchronous mode?
        if self.manager and self.manager.get_running_status() and hasattr(self, "world") and self.world:
            # Reset to asynchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
            self.traffic_manager.set_synchronous_mode(False)

        if self.manager:
            self.manager.cleanup()

        CarlaDataProvider.cleanup()

        for i, _ in enumerate(self.ego_vehicles):
            if self.ego_vehicles[i]:
                self.ego_vehicles[i].destroy()
                self.ego_vehicles[i] = None
        self.ego_vehicles = []

        if self._agent_watchdog:
            self._agent_watchdog.stop()

        if hasattr(self, "agent_instance") and self.agent_instance:
            self.agent_instance.destroy(results)
            del self.agent_instance

        if hasattr(self, "statistics_manager") and self.statistics_manager:
            self.statistics_manager.scenario = None

    def _prepare_ego_vehicles(self, ego_vehicles, wait_for_ego_vehicles=False):
        """
        Spawn or update the ego vehicles
        """

        if not wait_for_ego_vehicles:
            for vehicle in ego_vehicles:
                self.ego_vehicles.append(CarlaDataProvider.request_new_actor(vehicle.model, vehicle.transform, vehicle.rolename, color=vehicle.color, vehicle_category=vehicle.category)) # type: ignore

        else:
            ego_vehicle_missing = True
            while ego_vehicle_missing:
                self.ego_vehicles = []
                ego_vehicle_missing = False
                for ego_vehicle in ego_vehicles:
                    ego_vehicle_found = False
                    carla_vehicles = CarlaDataProvider.get_world().get_actors().filter("vehicle.*")
                    for carla_vehicle in carla_vehicles:
                        if carla_vehicle.attributes["role_name"] == ego_vehicle.rolename:
                            ego_vehicle_found = True
                            self.ego_vehicles.append(carla_vehicle)
                            break
                    if not ego_vehicle_found:
                        ego_vehicle_missing = True
                        break

            for i, _ in enumerate(self.ego_vehicles):
                self.ego_vehicles[i].set_transform(ego_vehicles[i].transform)

        # sync state
        CarlaDataProvider.get_world().tick()

    def _load_and_wait_for_world(self, args, town, ego_vehicles=None):
        """
        Load a new CARLA world and provide data to CarlaDataProvider
        """
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 1.0 / self.frame_rate
        settings.synchronous_mode = True
        self.world.apply_settings(settings)

        self.world.reset_all_traffic_lights()

        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)
        CarlaDataProvider.set_traffic_manager_port(int(args.trafficManagerPort))

        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(int(args.trafficManagerSeed))

        # Wait for the world to be ready
        if CarlaDataProvider.is_sync_mode():
            self.world.tick()
        else:
            self.world.wait_for_tick()

        if CarlaDataProvider.get_map().name.split("/")[-1] != town:
            raise Exception("The CARLA server uses the wrong map!" "This scenario requires to use map {}".format(town))

    def _register_statistics(self, config, route_date_string, checkpoint, entry_status, crash_message=""):
        """
        Computes and saved the simulation statistics
        """
        # register statistics
        current_stats_record = self.statistics_manager.compute_route_statistics(config, route_date_string, self.manager.scenario_duration_system, self.manager.scenario_duration_game, crash_message)

        print("\033[1m> Registering the route statistics\033[0m")
        self.statistics_manager.save_record(current_stats_record, config.index, checkpoint)
        self.statistics_manager.save_entry_status(entry_status, False, checkpoint)

        return current_stats_record

    def _load_and_run_scenario(self, args, config):
        """
        Load and run the scenario given by config.

        Depending on what code fails, the simulation will either stop the route and
        continue from the next one, or report a crash and stop.
        """
        crash_message = ""
        entry_status = "Started"

        os.environ["ROUTE_NAME"] = config.name
        print("\n\033[1m========= Preparing {} (repetition {}, index {}) =========".format(config.name, config.repetition_index, args.index), flush=True)
        print("> Setting up the agent\033[0m")

        # Prepare the statistics of the route
        self.statistics_manager.set_route(config.name, config.index)
        # Randomize during data collection.
        # Deterministic seed during evaluation.
        if int(os.environ.get("DATAGEN", 0)) == 1:
            CarlaDataProvider._rng = random.RandomState(seed=None)
        else:
            CarlaDataProvider._rng = random.RandomState(seed=config.index)

        now = datetime.now()
        route_string = pathlib.Path(os.environ.get("ROUTES", "")).stem + "_"
        route_string += f"route{config.index}"
        route_date_string = route_string + "_" + "_".join(map(lambda x: "%02d" % x, (now.month, now.day, now.hour, now.minute, now.second)))

        # Set up the user's agent, and the timer to avoid freezing the simulation
        try:
            self._agent_watchdog.start()
            agent_class_name = getattr(self.module_agent, "get_entry_point")()
            if int(os.environ.get("DATAGEN", 0)) == 1:
                self.agent_instance = getattr(self.module_agent, agent_class_name)(args.agent_config, config.index)
            elif agent_class_name == "SensorAgent":
                self.agent_instance = getattr(self.module_agent, agent_class_name)(args.agent_config, route_date_string, vehicle_config=self.vehicle_config)
            else:
                self.agent_instance = getattr(self.module_agent, agent_class_name)(args.agent_config, route_date_string)
            config.agent = self.agent_instance

            # Check and store the sensors
            if not self.sensors:
                self.sensors = self.agent_instance.sensors()
                track = self.agent_instance.track

                AgentWrapper.validate_sensor_configuration(self.sensors, track, args.track)

                self.sensor_icons = [sensors_to_icons[sensor["type"]] for sensor in self.sensors]
                self.statistics_manager.save_sensors(self.sensor_icons, args.checkpoint)

            self._agent_watchdog.stop()

        except SensorConfigurationInvalid as e:
            # The sensors are invalid -> set the ejecution to rejected and stop
            print("\n\033[91mThe sensor's configuration used is invalid:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Agent's sensors were invalid"
            entry_status = "Rejected"

            result = self._register_statistics(config, route_date_string, args.checkpoint, entry_status, crash_message)
            self._cleanup(result)
            return False
            # sys.exit(-1)
        
        except RuntimeError as e:
            traceback.print_exc()
            crash_message = "Agent took too long to setup"
            entry_status = "Crashed"
            
            result = self._register_statistics(config, route_date_string, args.checkpoint, entry_status, crash_message)
            self._cleanup(result)
            return False

        except Exception as e:
            # The agent setup has failed -> start the next route
            print("\n\033[91mCould not set up the required agent:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Agent couldn't be set up"

            result = self._register_statistics(config, route_date_string, args.checkpoint, entry_status, crash_message)
            self._cleanup(result)
            return False
        

        print("\033[1m> Loading the world\033[0m")

        # Load the world and the scenario
        try:
            self.world = self.client.load_world(config.town)
            print(f"Loaded map: {config.town}")
        except RuntimeError:
            # For cases where load_world crashes, but the world was properly
            # loaded anyways
            print(f"Cannot load map: {config.town}")
            self.world = self.client.get_world()
            return False
        try:
            self._load_and_wait_for_world(args, config.town, config.ego_vehicles)
            self._prepare_ego_vehicles(config.ego_vehicles, False)
            scenario = RouteScenario(world=self.world, vehicle_config=self.vehicle_config, config=config, debug_mode=args.debug, vehicle_index=args.index)
            self.statistics_manager.set_scenario(scenario.scenario)
            
            if int(os.environ.get("RANDOM_PHYSICS", 0)) == 1:
                self.agent_instance.update_physics()

            # Night mode
            if config.weather.sun_altitude_angle < 0.0:
                for vehicle in scenario.ego_vehicles:
                    vehicle.set_light_state(carla.VehicleLightState(self._vehicle_lights))

            # Load scenario and run it
            if args.record:
                self.client.start_recorder("{}/{}_rep{}.log".format(args.record, config.name, config.repetition_index))
            self.manager.load_scenario(scenario, self.agent_instance, config.repetition_index)

        except Exception as e:
            # The scenario is wrong -> set the ejecution to crashed and stop
            print("\n\033[91mThe scenario could not be loaded:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Simulation crashed"
            entry_status = "Crashed"

            result = self._register_statistics(config, route_date_string, args.checkpoint, entry_status, crash_message)

            if args.record:
                self.client.stop_recorder()

            self._cleanup(result)
            # sys.exit(-1)
            return False

        print("\033[1m> Running the route\033[0m")

        # Run the scenario
        try:
            self.manager.run_scenario()
            print("Scenario finished")

        except AgentError as e:
            # The agent has failed -> stop the route
            print("\n\033[91mStopping the route, the agent has crashed:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Agent crashed"

        except Exception as e:
            print("\n\033[91mError during the simulation:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Simulation crashed"
            entry_status = "Crashed"

        # Stop the scenario
        try:
            print("\033[1m> Stopping the route\033[0m")
            self.manager.stop_scenario()
            result = self._register_statistics(config, route_date_string, args.checkpoint, entry_status, crash_message)

            if args.record:
                self.client.stop_recorder()

            # Remove all actors
            scenario.remove_all_actors()

            self._cleanup(result)

        except Exception as e:
            print("\n\033[91mFailed to stop the scenario, the statistics might be empty:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Simulation crashed"

        if crash_message == "Simulation crashed":
            return False
        else:
            return True
    
    def check_resume(self, args, index, success_list, data):
        try:
            log = data["_checkpoint"]["global_record"]["meta"]["exceptions"][index][2]
            resume = (int(args.resume) == 1 and int(args.resume_failed) == 1 and log in FAILED_LOG) or (success_list[index] == False)
        except Exception:
            resume = True
            log = None
        return resume, log

    def run(self, args, route_indexer, success_list):
        result = True
        run = False
        data = fetch_dict(args.checkpoint)
        
        while route_indexer.peek():
            resume, log = self.check_resume(args, route_indexer._index, success_list, data)
            if resume:
                if log is not None:
                    print(f"Resume: RouteScenario_{route_indexer._index}: {log}")
                else:
                    print("Starting new route.")
                try:
                    run = True
                    # setup
                    config = route_indexer.next()
                    print("Load and run scenarios.")

                    # run
                    result = self._load_and_run_scenario(args, config)

                    print("Save state.")
                    route_indexer.save_state(args.checkpoint)

                    success_list[route_indexer._index - 1] = result
                    with open(args.result_list, "wb") as f:
                        pickle.dump(success_list, f)
                        
                except RuntimeError as e:
                    self.global_statistics(args, route_indexer)
                    print(f"RuntimeError - Retry: {e}")
                    traceback.print_exc()
                    return False
                except KeyboardInterrupt as e:
                    self.global_statistics(args, route_indexer)
                    print(f"KeyboardInterrupt - Retry: {e}")
                    traceback.print_exc()
                    return False
                if run:
                    self.global_statistics(args, route_indexer)
            else:
                print(f"Skip: RouteScenario_{route_indexer._index}: {log}")
                config = route_indexer.next()
                result = True
            
        return result

    def global_statistics(self, args, route_indexer):
        # save global statistics
        print("\033[1m> Registering the global statistics\033[0m")
        global_stats_record = self.statistics_manager.compute_global_statistics(route_indexer.total)
        StatisticsManager.save_global_record(global_stats_record, self.sensor_icons, route_indexer.total, args.checkpoint)


def argument_parser():
    description = "CARLA AD Leaderboard Evaluation: evaluate your Agent in CARLA scenarios\n"

    # general parameters
    parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument("--host", default="localhost", help="IP of the host server (default: localhost)")
    parser.add_argument("--port", default="2000", help="TCP port to listen to (default: 2000)")
    parser.add_argument("--trafficManagerPort", default="8000", help="Port to use for the TrafficManager (default: 8000)")
    parser.add_argument("--trafficManagerSeed", default="0", help="Seed used by the TrafficManager (default: 0)")
    parser.add_argument("--debug", type=int, help="Run with debug output", default=0)
    parser.add_argument("--record", type=str, default="", help="Use CARLA recording feature to create a recording of the scenario")
    parser.add_argument("--timeout", default="60.0", help="Set the CARLA client timeout value in seconds")

    # simulation setup
    parser.add_argument("--routes", help="Name of the route to be executed. Point to the route_xml_file to be executed.", required=True)
    parser.add_argument("--scenarios", help="Name of the scenario annotation file to be mixed with the route.", required=True)
    parser.add_argument("--repetitions", type=int, default=1, help="Number of repetitions per route.")

    # agent-related options
    parser.add_argument("-a", "--agent", type=str, help="Path to Agent's py file to evaluate", required=True)
    parser.add_argument("--agent-config", type=str, help="Path to Agent's configuration file", default="")

    parser.add_argument("--track", type=str, default="SENSORS", help="Participation track: SENSORS, MAP")
    parser.add_argument("--resume", type=int, default=0, help="Resume execution from last checkpoint?")
    parser.add_argument("--resume-failed", type=int, default=0, help="Resume execution of failed scenarios?")
    parser.add_argument("--checkpoint", type=str, default="./simulation_results.json", help="Path to checkpoint used for saving statistics and resuming")

    parser.add_argument("--result-list", type=str, default="./result_list.pickle", help="List of results of the scenarios")
    parser.add_argument("--index", "-i", type=int, default=0, help="Index of the vehicle configuration to use")

    return parser.parse_args()


def main(args):
    """
    Run the challenge mode
    """
    statistics_manager = StatisticsManager()
    route_indexer = RouteIndexer(args.routes, args.scenarios, args.repetitions)
    checkpoint_dir = os.path.dirname(args.checkpoint)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(args.checkpoint):
        with open(args.checkpoint, "w") as f:
            json.dump({}, f)
    if not os.path.exists(args.result_list):
        success_list = [False] * len(route_indexer._configs_list)
        with open(args.result_list, "wb") as f:
            pickle.dump(success_list, f)

    if int(args.resume) == 0:
        statistics_manager.clear_record(args.checkpoint)
        route_indexer.save_state(args.checkpoint)
        success_list = [False] * len(route_indexer._configs_list)
    else:
        with open(args.result_list, "rb") as f:
            success_list = pickle.load(f)
        statistics_manager.resume(args.checkpoint)
        if int(args.resume_failed) == 0:
            route_indexer.resume(args.checkpoint)
        
    leaderboard_evaluator = LeaderboardEvaluator(args, statistics_manager)
    
    try: 
        result = leaderboard_evaluator.run(args, route_indexer, success_list)
    except KeyError as e:
        print(f"KeyError: {e}")
        result = False

    del leaderboard_evaluator
    
    if os.getenv("STREAM", 0):
        pygame.quit()
    with open(args.result_list, "wb") as f:
        pickle.dump(success_list, f)
    return result


if __name__ == "__main__":
    args = argument_parser()
    result = main(args)
    if result:
        print("Finished: Success")
        sys.exit(0)
    else:
        print("Finished: Failed")
        sys.exit(-1)
