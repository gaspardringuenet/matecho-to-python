#! /bin/bash

echo "ABRACOS A1"
matecho-convert ~/Data/tapioca/abracos/a1/acoustic/EI002_3pings1m/Echointegration.mat --light

echo "ABRACOS A2"
matecho-convert ~/Data/tapioca/abracos/a2/acoustic/EI002_3pings1m/Echointegration.mat --light

echo "AMAZOMIX"
matecho-convert ~/Data/tapioca/amazomix/acoustic/EI002_3pings1m/Echointegration_leg01.mat \
    ~/Data/tapioca/amazomix/acoustic/EI002_3pings1m/Echointegration_leg02A.mat \
    ~/Data/tapioca/amazomix/acoustic/EI002_3pings1m/Echointegration_leg02B.mat \
    --light
echo "RENAMING AMAZOMIX ZARR STORE"
rm -rf ~/Data/tapioca/amazomix/acoustic/EI002_3pings1m/Echointegration.zarr
mv ~/Data/tapioca/amazomix/acoustic/EI002_3pings1m/Echointegration_leg01.zarr \
    ~/Data/tapioca/amazomix/acoustic/EI002_3pings1m/Echointegration.zarr
