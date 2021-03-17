# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt


run_locally:
	@python -m ${PACKAGE_NAME}.${FILENAME}

check_code:
	@flake8 scripts/* ev_Stations_Model/*.py

black:
	@black scripts/* ev_Stations_Model/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr ev_Stations_Model-*.dist-info
	@rm -fr ev_Stations_Model.egg-info

install:
	@pip install . -U

all: clean install test black check_code


uninstal:
	@python setup.py install --record files.txt
	@cat files.txt | xargs rm -rf
	@rm -f files.txt

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#           GOOGLE CLOUD
# ----------------------------------

PROJECT_ID=ev-stations-202103
BUCKET_NAME=ev-stations-bucket-202103
REGION=europe-west1

######### Google Storage

LOCAL_PATH="/Users/nhuyghe/code/nicolashuyghe/ev_Stations_Model/raw_data/ytrain_raw.csv"
BUCKET_FOLDER=raw_data
BUCKET_FILE_NAME=$(shell basename ${LOCAL_PATH})

set_project:
	@gcloud config set project ${PROJECT_ID}

create_bucket:
	@gsutil mb -l ${REGION} -p ${PROJECT_ID} gs://${BUCKET_NAME}

upload_data:
	@gsutil cp ${LOCAL_PATH} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${BUCKET_FILE_NAME}


######### AI Platform

BUCKET_TRAINING_FOLDER=training
PACKAGE_NAME=ev_Stations_Model
FILENAME=model
PYTHON_VERSION=3.7
RUNTIME_VERSION=2.3

JOB_NAME=ev_stations_training_model_$(shell date +'%Y%m%d_%H%M%S')

gcp_submit_training:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--stream-logs

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)

# ----------------------------------
#              RUN API
# ----------------------------------
run_api:
	uvicorn api.fast:app --reload  # load web server with code autoreload
