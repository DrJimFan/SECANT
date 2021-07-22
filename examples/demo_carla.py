#!/usr/bin/env python
import argparse
import os
import time
from pprint import pformat
import socket
from secant.envs.carla import make_carla
import subprocess as sp
import psutil


def random_free_tcp_port() -> int:
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(("", 0))
    addr, port = tcp.getsockname()
    tcp.close()
    return port


parser = argparse.ArgumentParser()
parser.add_argument(
    "-p",
    "--port",
    type=int,
    default=-1,
    help="If port == -1, randomly select a free port to use",
)
parser.add_argument("-m", "--map", type=int, default=1, help="town0x map, from 1 to 5")
parser.add_argument(
    "-ne", "--no-server", action="store_true", help="Do not launch a server subprocess"
)
parser.add_argument(
    "-e",
    "--server-executable",
    type=str,
    default="/usr/local/bin/carla-server",
    help="town0x map, from 1 to 5",
)
args = parser.parse_args()
assert 1 <= args.map <= 5
if args.port < 0:
    args.port = random_free_tcp_port()

if not args.no_server:
    proc = sp.Popen(
        f"DISPLAY= {args.server_executable} "
        f"-opengl -nosound -carla-world-port={args.port}",
        shell=True,
    )
    print(f"Server starting at port {args.port}, wait up to 15 seconds ...")
    time.sleep(15)
    print("Server process ID:", proc.pid)
else:
    proc = None

print(f"Launching client to connect to port {args.port} ...")
env = make_carla(
    f"town0{args.map}", client_port=args.port, modalities=["rgb", "gnss"]
)
print("Created env successfully.")

print(env.reset())

obs, reward, done, info = env.step(env.action_space.sample())

print("Reward=", reward)
print("Info=", pformat(info))

env.close()
print("Client exits successfully.")

if proc is not None:
    # carla also launches UE4 subprocess, needs to kill recursively
    for child in psutil.Process(proc.pid).children(recursive=True):
        child.kill()
    proc.kill()
    print("Server exits successfully.")
