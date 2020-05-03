NAME=vehicles-ml-api

build-ml-api-aws:
	docker build --build-arg PIP_REMOTE_PACKAGE=${PIP_REMOTE_PACKAGE} --build-arg TRUSTED_HOST=${TRUSTED_HOST} --build-arg AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} --build-arg AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} -t $(NAME):latest .

push-ml-api-aws:
	docker push ${AWS_ACCOUNT_ID}.dkr.ecr.us-west-1.amazonaws.com/$(NAME):latest

tag-ml-api:
	docker tag $(NAME):latest ${AWS_ACCOUNT_ID}.dkr.ecr.us-west-1.amazonaws.com/$(NAME):latest
