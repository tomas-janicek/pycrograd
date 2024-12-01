# Timed Training Test

Tested with: `PYTHONPATH=. python pycrograd/cli.py run_digits_benchmark --epochs=1 --length=10` with 1 epochs and length set to 10 for digits dataset in batches of 32 examples.

### MacBook M1

#### Network size 64 -> 64 -> 32 -> 10

Training without any optimization: 0.11281895637512207 seconds

#### Network size 64 [(-> 256) * 29] -> 10 (total of 30 layers)

Training without any optimization: 38.23866581916809 seconds

#### Network size 64 -> 8192 -> 4096 -> 2048 -> 10

Training without any optimization: 916.3234758377075 seconds
