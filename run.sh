#!/bin/bash
echo "Starting FastAPI application..."
gunicorn -c gunicorn_config.py app.main:app