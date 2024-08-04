import ray
ray.init(address='auto')


@ray.remote
def f(x):
    return x*x

futures = [f.remote(i) for i in range(200000)]

results = ray.get(futures)
print(results)