serve:
	docker compose --profile serve up -d
train:
	docker compose --profile train up --abort-on-container-exit
train-serve:
	docker compose --profile train up
down:
	docker compose --profile train --profile serve down