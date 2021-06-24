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
