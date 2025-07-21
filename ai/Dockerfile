FROM public.ecr.aws/lambda/python:3.13

WORKDIR /code

ENV LAMBDA_TASK_ROOT=/code

COPY ./requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app ${LAMBDA_TASK_ROOT}/app

CMD ["app.script.lambda_handler"]
