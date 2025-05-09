#!/bin/bash

DATAPATH=$1

echo "$DATAPATH"

JSON_STRING=$(jq -n \
		 --arg dp "$DATAPATH" \
		 '
{
        "app": {
            "repository": "/data1/beniz/code/colette_v2/apps/coder",
            "verbose": "debug"
        },
        "parameters": {
            "input": {
            },
            "llm": {
                "source": "qwen2.5-coder",
                "inference": {
                    "lib": "ollama"
                },
		"conversational": true
            }
        }
    }
'
	   )

#echo $JSON_STRING

curl -X PUT -H 'Content-Type: application/json' http://localhost:1873/v2/app/coder -d "$JSON_STRING"
