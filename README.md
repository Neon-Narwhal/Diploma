# Diploma

## old
sh run_a100.sh

## Docker

docker build -t your-registry/diploma_bench:latest .
docker push your-registry/diploma_bench:latest


Узнать ID контейнера (пока он еще жив или уже умер, но не удален)
docker cp <container_id>:/app/ml/outputs ./my_results


