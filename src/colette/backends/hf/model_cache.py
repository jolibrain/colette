import threading


class ModelCache:
    _cache = {}

    @classmethod
    def get(cls, key):
        if key in cls._cache:
            cls._cache[key]["ref_count"] += 1
            return cls._cache[key]["model"]
        return None

    @classmethod
    def info(cls, key):
        if key in cls._cache:
            return cls._cache[key]["ref_count"]
        return -1

    @classmethod
    def add(cls, key, model, processor, llm_type):
        cls._cache[key] = {
            "model": (model, processor, llm_type),
            "ref_count": 1,
            "lock": threading.Lock(),
        }

    @classmethod
    def acquire_lock(cls, key):
        if key in cls._cache:
            cls._cache[key]["lock"].acquire()

    @classmethod
    def release_lock(cls, key):
        if key in cls._cache:
            cls._cache[key]["lock"].release()

    @classmethod
    def release(cls, key):
        if key in cls._cache:
            cls._cache[key]["ref_count"] -= 1
            if cls._cache[key]["ref_count"] == 0:
                del cls._cache[key]

    @classmethod
    def is_in_use(cls, key, c):
        return key in cls._cache and cls._cache[key]["ref_count"] > c
