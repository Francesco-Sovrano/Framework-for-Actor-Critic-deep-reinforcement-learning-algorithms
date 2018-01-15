#!/bin/bash

cd ./log
. .env/bin/activate
../A3C/train.sh > out.log &
disown
exit
