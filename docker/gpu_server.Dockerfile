ARG GTAG=latest
FROM docker.jolibrain.com/colette_gpu:$GTAG AS colette_gpu_server
WORKDIR /app

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/app/server/run.sh", "--host", "0.0.0.0", "--port", "1873"]
