## convert binary file in CSV text format
## the binary file has header array and value array
## in the header array there are dimensions (X and Y size)
## in value array there are Z's values

import argparse

import numpy as np
from datetime import date

from byteorder import int32,float32

parser = argparse.ArgumentParser(
    description="easy way to convert CSV in binary")

parser.add_argument("--csv-path", 
                    help="path of CSV to read")

parser.add_argument("--binary-path", 
                    help="path of binary to save",
                    default=date.today().strftime("%m-%d-%Y_%H-%M-%S") + "_binary_csv.bin")

parser.add_argument("--dim-profile", 
                    type=int, default=4096,
                    help="number of X points in one profile, DEFAULT=4096, TYPE=int")

parser.add_argument("--num-profile", 
                    type=int,
                    help="number profile in the scan, TYPE=int")

args = parser.parse_args()

## read e convert binary file
print(" --- start convert binary file ---")

with open(args.binary_path, mode='rb') as i:
    value_0 = np.fromfile(i, dtype=int32, count=2)
    value_1 = np.fromfile(i, dtype=float32, count=value_0[0]*value_0[1])

print(" --- end conversion ---")

print("--- start write ---")

## open stream
stream_write = open(args.csv_path, 'w')

## write in append to filw
value_0.tofile(stream_write)
value_1.tofile(stream_write)

## close file
stream_write.close()
print("--- end write ---")