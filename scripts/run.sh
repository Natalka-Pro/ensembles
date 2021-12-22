#!/bin/bash

sudo docker run -i -t --rm -p 5000:5000 -v "$PWD/server/data:/root/server/data" --rm -i repo_name/server
