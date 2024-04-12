set -e

# Prerequisite: aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 097863529862.dkr.ecr.us-west-2.amazonaws.com

docker build -t llmchan-webui .
docker tag llmchan-webui:latest 097863529862.dkr.ecr.us-west-2.amazonaws.com/llmchan-webui:latest
docker push 097863529862.dkr.ecr.us-west-2.amazonaws.com/llmchan-webui:latest