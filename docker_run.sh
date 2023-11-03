#!/bin/bash

IMAGE="digit:v1"

#creating volume
docker volume create models

#building docker
docker build -t $IMAGE -f docker/Dockerfile .
docker run -d --name MODELS -v models:/digits/models $IMAGE
docker cp MODELS:/digits/models/ \\wsl.localhost\Ubuntu-22.04\home\dipan\mlops_23