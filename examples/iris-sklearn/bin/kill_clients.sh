docker kill $(docker ps -aq --filter "name=client")
docker rm $(docker ps -aq --filter "name=client")
docker rmi $(docker images -aq --filter "reference=iris-sklearn:*")
sudo rm -rf data/clients/