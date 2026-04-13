#!/bin/sh

# Run database migrations before starting the server.
# This is the only place migrations execute — the proxy starts with
# DISABLE_SCHEMA_UPDATE=true so it never auto-migrates on its own.
# Running here (not inside the Python process) keeps the migration step
# explicit and prevents the litellm_proxy_extras _resolve_all_migrations
# diffing logic from touching our custom tables.
if [ -n "$DATABASE_URL" ]; then
    echo "Running custom DB migrations..."
    prisma migrate deploy --schema /app/litellm/proxy/schema.prisma
    if [ $? -ne 0 ]; then
        echo "ERROR: DB migration failed, aborting startup"
        exit 1
    fi
fi

if [ "$SEPARATE_HEALTH_APP" = "1" ]; then
    export LITELLM_ARGS="$@"
    export SUPERVISORD_STOPWAITSECS="${SUPERVISORD_STOPWAITSECS:-3600}"
    exec supervisord -c /etc/supervisord.conf
fi

if [ "$USE_DDTRACE" = "true" ]; then
    export DD_TRACE_OPENAI_ENABLED="False"
    exec ddtrace-run litellm "$@"
else
    exec litellm "$@"
fi