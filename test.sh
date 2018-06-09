#!/bin/bash

k=$(wget -qO- http://localhost:9009/detect/image/river)
echo $k
for i in $k; do
    echo $i
done
