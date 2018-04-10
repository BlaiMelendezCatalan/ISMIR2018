NAME=blaimelcat/modified_tsipas_alg

build:
	docker build -t $(NAME) .

run:
	docker run -it --rm=true $(NAME) bash

remove:
	-docker stop $(NAME)
	-docker rm $(NAME)

env:
	virtualenv env
	. env/bin/activate && pip install -r requirements.txt
