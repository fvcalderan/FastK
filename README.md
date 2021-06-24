# FastK
Python implementation of FastK K-Medoids clustering algorithm (except it's slow).

I wanted to learn how K-Medoids works, looked it up on the internet and found
[Park and Jun's paper](https://doi.org/10.1016/j.eswa.2008.01.039), which has over 1500
citations as of June 6th, 2021. This particular implementation of mine is very
unoptimized and was made just for learning purposes. In the future, I might
implement this algorithm in C and try to do some fun multithreading
optimizations.

FastK itself is as class inside the file `FastK.py`. Its fit function receives two parameters: the dataset to be optimized and the number K of clusters to be generated. The dataset can be anything that can be turned into a NumPy array, such as a `numpy.ndarray` or `pandas.DataFrame`. There's an example program inside `example.py` and since it's so small, here it is in its full length:

```python
import time
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from FastK import FastK


def main():
    # Create dataset
    df = make_blobs(n_samples=100, centers=4, n_features=2,
                    cluster_std=0.6, random_state=50)[0]

    # Cluster (and measure the time taken)
    fastk = FastK()
    start_t = time.time()
    fastk.fit(df, n_clusters=4)
    end_t = time.time()

    # Output time taken and scatter plot
    print(f'Time taken: {end_t-start_t}s')
    x, y = list(zip(*df))
    plt.scatter(x, y, c=fastk.labels_)
    plt.show()


if __name__ == '__main__':
    main()
```

Have fun c: