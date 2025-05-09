### Steps

1 - step 1: start the server

```bash
CUDA_VISIBLE_DEVICES=3 server/run.sh --host 0.0.0.0
```

2 - step 2: create the app

```bash
curl -X POST "http://localhost:1873/v1/index/app_colette" -H "Content-Type: application/json" -d @create_app.json
```

3 - step 3: index the files

```bash
curl -X PUT "http://localhost:1873/v1/index/app_colette" -H "Content-Type: application/json" -d @index_app.json
```

4 - step 4: send a query

```bash
curl -X POST "http://localhost:1873/v1/predict/app_colette" -H "Content-Type: application/json" -d @query_app.json
```
