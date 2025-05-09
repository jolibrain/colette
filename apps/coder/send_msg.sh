#!/bin/bash

MSG="what is the number of space debris ?"

curl -X POST -H 'Content-Type: application/json' http://localhost:1873/v2/predict/coder -d '
{  
    "app": {
        "verbose": "debug"
    },
    "parameters": {
        "input": {
            "message": "Write a Python program that implements the Hungarian algorithm."
        }
    }
}
'
