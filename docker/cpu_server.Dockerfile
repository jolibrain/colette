ARG GTAG=latest
from colette_cpu_build:$GTAG AS colette_cpu_server
WORKDIR /app
ENTRYPOINT ["/app/server/run.sh", "--host", "0.0.0.0", "--port", "1873"]