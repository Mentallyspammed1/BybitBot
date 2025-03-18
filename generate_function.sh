#!/bin/bash

    LANGUAGE="$1"
    DESCRIPTION="$2"
    PARAMETERS="$3"  # Optional
    RETURN_TYPE="$4" # Optional                                     ERROR_HANDLING="$5" # Optional, boolean
    COMMENTS="$6" # Optional, boolean
                                                                    if [ -z "$LANGUAGE" ] || [ -z "$DESCRIPTION" ]; then
      echo "Error: Language and description are required."            exit 1
    fi                                                          
    PROMPT="Generate a function in ${LANGUAGE} that does the    following: ${DESCRIPTION}."
                                                                    if [ -n "$PARAMETERS" ]; then
      PROMPT="${PROMPT} Function parameters should be:          ${PARAMETERS}."
    fi                                                              if [ -n "$RETURN_TYPE" ]; then
      PROMPT="${PROMPT} The function should return a            ${RETURN_TYPE}."
    fi                                                              if [[ "$ERROR_HANDLING" == "true" ]]; then
      PROMPT="${PROMPT} Include basic error handling for invalidinputs."
    fi                                                              if [[ "$COMMENTS" == "true" ]]; then
      PROMPT="${PROMPT} Add comments to explain the code."          fi
                                                                    PROMPT="${PROMPT} Please provide only the function code,
without surrounding text or explanations."                      
                                                                    GENERATED_CODE=$(aichat --text "$PROMPT")
                                                                    if [ -n "$GENERATED_CODE" ]; then
      echo "$GENERATED_CODE"                                        else
      echo "Error: Could not generate function code."                 exit 1
    fi
