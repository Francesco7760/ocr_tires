## convert CSV in binary LV

## the CSV file has 3 column (X,Y and Z) we want convert in binary format
## third column, so we extract Z and convert

import argparse

import numpy as np
import pandas as pd
from datetime import date

from byteorder import int32,float32

def read_array(path):

    ## work with csv
    print("--- start read ---")
    df = pd.read_csv(path, sep=",",header=None, low_memory=False)
    print("--- end read ---")

    return df

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

## read CSV
df = read_array(args.csv_path)

## get only columns of Z and rashape vector to take correct matrix
array = df[[2]].iloc[1:].to_numpy().reshape(args.num_profile,args.dim_profile)

## convert array in 2 object 
## - size array
## - values array 
print("--- start convert binary ---")
size_array = np.array([array.shape[0],array.shape[1]], dtype=int32)
#print(size_array)
values_array = np.array(array.reshape(array.shape[0]*array.shape[1]),dtype=float32)
#print(values_array)
print("--- end conversion ---")

print("--- start write ---")

## open stream
stream_write = open(args.binary_path, 'w')

## write in append to filw
size_array.tofile(stream_write)
values_array.tofile(stream_write)

## close file
stream_write.close()
print("--- end write ---")