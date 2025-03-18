#!/usr/bin/env bash

# @describe Join lines with commas
# @option -t --text! Input text (newline-separated)

function run() {
  echo "$argc_text" | tr '\n' ',' | sed 's/,$//'
}

eval "$(argc --argc-eval "$0" "$@")"
