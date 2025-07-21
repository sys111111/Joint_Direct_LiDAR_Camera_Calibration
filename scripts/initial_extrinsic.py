#!/usr/bin/env python3

import argparse
import json
import os
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Update the 'results.init_T_lidar_camera' entry in calib.json")
    parser.add_argument(
        "extrinsics",
        help="Seven numbers in order: tx,ty,tz,qx,qy,qz,qw"
    )
    parser.add_argument(
        "--data_path",
        required=True,
        help="Directory containing calib.json"
    )

    args = parser.parse_args()

    parts = args.extrinsics.split(",")
    if len(parts) != 7:
        print("Error: 'extrinsics' must contain exactly 7 comma-separated values", file=sys.stderr)
        sys.exit(1)

    try:
        values = [float(p) for p in parts]
    except ValueError:
        print("Error: all 7 values must be valid floats", file=sys.stderr)
        sys.exit(1)

    calib_path = os.path.join(args.data_path, "calib.json")
    if not os.path.isfile(calib_path):
        print(f"Error: {calib_path} not found", file=sys.stderr)
        sys.exit(1)

    # Load the existing JSON
    with open(calib_path, "r") as f:
        config = json.load(f)

    # Ensure 'results' exists, update the field
    if "results" not in config:
        config["results"] = {}
    config["results"]["init_T_lidar_camera"] = values

    # Write back to file
    with open(calib_path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    print("Successfully updated init_T_lidar_camera to:")
    print(values)

if __name__ == "__main__":
    main()
