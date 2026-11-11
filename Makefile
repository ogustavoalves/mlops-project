# Run just the server.
serve:
	docker compose --profile serve up -d
# Run just the training script.
train:
	docker compose --profile train up --abort-on-container-exit
# Run training and server.
train-serve:
	docker compose --profile train up
# Stops training and/or server.
down:
	docker compose --profile train --profile serve down
# Builds server and train images. Don't run any container.
build:
	docker compose --profile train --profile serve build 