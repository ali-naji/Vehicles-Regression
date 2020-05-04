FROM python:3.6.4

RUN adduser --disabled-password --gecos '' ml-api-user

WORKDIR /opt/ml_api

# copying dir to image 
ADD packages/ml_api /opt/ml_api

# setting variables and build arguments
ARG PIP_REMOTE_PACKAGE
ARG TRUSTED_HOST
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ENV FLASK_APP run.py

# installing requirements
ADD ./packages/ml_api /opt/ml_api/
RUN pip install --upgrade pip
RUN pip install -r /opt/ml_api/requirements.txt

# give permission and ownership to user
RUN chmod +x /opt/ml_api/run.sh
RUN chown -R ml-api-user:ml-api-user ./

USER ml-api-user

# setup the port and start hosting
EXPOSE 5000
CMD ["bash", "./run.sh"]