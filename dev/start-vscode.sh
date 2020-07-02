#!/bin/bash
set -e
PASSWORD=$VSCODE_TOKEN /usr/local/bin/start.sh nohup code-server --bind-addr 0.0.0.0:8887 \
$WORK_DIR/dev.code-workspace > $WORK_DIR/logs/code-server.log 2>&1 &
