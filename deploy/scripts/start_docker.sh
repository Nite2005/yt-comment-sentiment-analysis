#!/bin/bash
# Log everything to start_docker.log
exec > /home/ubuntu/start_docker.log 2>&1

echo "Logging in to ECR..."
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 619071355884.dkr.ecr.ap-south-1.amazonaws.com
echo "Pulling Docker image..."
docker pull 619071355884.dkr.ecr.ap-south-1.amazonaws.com/yt-comment:latest

echo "Checking for existing container..."
if [ "$(docker ps -q -f name=yt-chrome-plugin)" ]; then
    echo "Stopping existing container..."
    docker stop yt-chrome-plugin
fi

if [ "$(docker ps -aq -f name=yt-chrome-plugin)" ]; then
    echo "Removing existing container..."
    docker rm yt-chrome-plugin
fi

echo "Starting new container..."
docker run -d -p 80:5000 --name yt-chrome-plugin 619071355884.dkr.ecr.ap-south-1.amazonaws.com/yt-comment:latest

echo "Container started successfully."