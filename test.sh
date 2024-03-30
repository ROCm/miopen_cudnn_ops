#!/usr/bin/env bash

./rebuild.sh
echo

echo './build/op_driver rnn     -n 224 -W 224 -H 1000 -l 8 -b 1 -m lstm -p 0 -r 0 -k 32 -c 0 -F 0 -t 1 -w 1'
./build/op_driver rnn     -n 224 -W 224 -H 1000 -l 8 -b 1 -m lstm -p 0 -r 0 -k 32 -c 0 -F 0 -t 1 -w 1
echo

echo './build/op_driver rnnfp16 -n 224 -W 224 -H 1000 -l 8 -b 1 -m lstm -p 0 -r 0 -k 32 -c 0 -F 0 -t 1 -w 1'
./build/op_driver rnnfp16 -n 224 -W 224 -H 1000 -l 8 -b 1 -m lstm -p 0 -r 0 -k 32 -c 0 -F 0 -t 1 -w 1
echo
