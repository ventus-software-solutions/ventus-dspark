# Dashboard image — stdlib Python only, so there is nothing to install and
# nothing to audit beyond server.py itself.
FROM python:3.12-alpine

RUN adduser -D -u 10001 dash
WORKDIR /app
COPY server.py /app/server.py
USER dash

EXPOSE 8500
CMD ["python3", "-u", "/app/server.py"]
