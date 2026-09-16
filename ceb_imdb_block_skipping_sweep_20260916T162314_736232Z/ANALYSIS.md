# CEB IMDb block-skipping sweep

All 153 evaluated parameter combinations skipped **0 blocks** (`total_num_skipped_blocks = 0`). Equivalently, each combination pruned none of its 29,450 candidate query/block pairs. The mean unnecessary-block-read ratio is therefore `1.0` throughout the sweep.

Widths are grouped below because every listed width has the same result.

| Feature selection | n-gram | Min block frequency | Max block frequency | Fingerprint widths (bits) | Blocks skipped |
|---|---:|---:|---:|---|---:|
| `local_split_entropy` | 2 | 0 | 0.1 | 8, 16, 32, 64, 128, 256 | 0 at every width |
| `local_split_entropy` | 2 | 0 | 0.25 | 8, 16, 32, 64, 128, 256 | 0 at every width |
| `local_split_entropy` | 2 | 0 | 0.5 | 8, 16, 32, 64, 128, 256 | 0 at every width |
| `local_split_entropy` | 2 | 0 | 1 | 8, 16, 32, 64, 128, 256 | 0 at every width |
| `local_split_entropy` | 3 | 0 | 0.1 | 8, 16, 32, 64, 128, 256 | 0 at every width |
| `local_split_entropy` | 3 | 0 | 0.25 | 8, 16, 32, 64, 128, 256 | 0 at every width |
| `local_split_entropy` | 3 | 0 | 0.5 | 8, 16, 32, 64, 128, 256 | 0 at every width |
| `local_split_entropy` | 3 | 0 | 1 | 8, 16, 32, 64, 128, 256 | 0 at every width |
| `local_split_entropy` | 4 | 0 | 0.1 | 8, 16, 32, 64, 128, 256 | 0 at every width |
| `local_split_entropy` | 4 | 0 | 0.25 | 8, 16, 32, 64, 128, 256 | 0 at every width |
| `local_split_entropy` | 4 | 0 | 0.5 | 8, 16, 32, 64, 128, 256 | 0 at every width |
| `local_split_entropy` | 4 | 0 | 1 | 8, 16, 32, 64, 128, 256 | 0 at every width |
| `fingerprint_distribution_entropy` | 2 | 0 | 0.1 | 8, 16, 32 | 0 at every width |
| `fingerprint_distribution_entropy` | 2 | 0 | 0.25 | 8, 16 | 0 at every width |
| `fingerprint_distribution_entropy` | 2 | 0 | 0.5 | 8, 16 | 0 at every width |
| `fingerprint_distribution_entropy` | 2 | 0 | 1 | 8, 16 | 0 at every width |
| `fingerprint_distribution_entropy` | 2 | 0.01 | 0.1 | 8, 16, 32 | 0 at every width |
| `fingerprint_distribution_entropy` | 2 | 0.01 | 0.25 | 8, 16 | 0 at every width |
| `fingerprint_distribution_entropy` | 2 | 0.01 | 0.5 | 8, 16 | 0 at every width |
| `fingerprint_distribution_entropy` | 2 | 0.01 | 1 | 8, 16 | 0 at every width |
| `fingerprint_distribution_entropy` | 2 | 0.05 | 0.1 | 8, 16, 32 | 0 at every width |
| `fingerprint_distribution_entropy` | 2 | 0.05 | 0.25 | 8, 16 | 0 at every width |
| `fingerprint_distribution_entropy` | 2 | 0.05 | 0.5 | 8, 16 | 0 at every width |
| `fingerprint_distribution_entropy` | 2 | 0.05 | 1 | 8, 16 | 0 at every width |
| `fingerprint_distribution_entropy` | 3 | 0 | 0.1 | 8, 16, 32 | 0 at every width |
| `fingerprint_distribution_entropy` | 3 | 0 | 0.25 | 8, 16 | 0 at every width |
| `fingerprint_distribution_entropy` | 3 | 0 | 0.5 | 8, 16 | 0 at every width |
| `fingerprint_distribution_entropy` | 3 | 0 | 1 | 8, 16 | 0 at every width |
| `fingerprint_distribution_entropy` | 3 | 0.01 | 0.1 | 8, 16, 32 | 0 at every width |
| `fingerprint_distribution_entropy` | 3 | 0.01 | 0.25 | 8, 16 | 0 at every width |
| `fingerprint_distribution_entropy` | 3 | 0.01 | 0.5 | 8, 16 | 0 at every width |
| `fingerprint_distribution_entropy` | 3 | 0.01 | 1 | 8, 16 | 0 at every width |
| `fingerprint_distribution_entropy` | 3 | 0.05 | 0.1 | 8, 16, 32 | 0 at every width |
| `fingerprint_distribution_entropy` | 3 | 0.05 | 0.25 | 8, 16 | 0 at every width |
| `fingerprint_distribution_entropy` | 3 | 0.05 | 0.5 | 8, 16 | 0 at every width |
| `fingerprint_distribution_entropy` | 3 | 0.05 | 1 | 8, 16 | 0 at every width |
| `fingerprint_distribution_entropy` | 4 | 0 | 0.1 | 8, 16, 32 | 0 at every width |
| `fingerprint_distribution_entropy` | 4 | 0 | 0.25 | 8, 16 | 0 at every width |
| `fingerprint_distribution_entropy` | 4 | 0 | 0.5 | 8, 16 | 0 at every width |
| `fingerprint_distribution_entropy` | 4 | 0 | 1 | 8, 16 | 0 at every width |
| `fingerprint_distribution_entropy` | 4 | 0.01 | 0.1 | 8, 16, 32 | 0 at every width |
| `fingerprint_distribution_entropy` | 4 | 0.01 | 0.25 | 8, 16 | 0 at every width |
| `fingerprint_distribution_entropy` | 4 | 0.01 | 0.5 | 8, 16 | 0 at every width |
| `fingerprint_distribution_entropy` | 4 | 0.01 | 1 | 8, 16 | 0 at every width |
| `fingerprint_distribution_entropy` | 4 | 0.05 | 0.1 | 8, 16, 32 | 0 at every width |
| `fingerprint_distribution_entropy` | 4 | 0.05 | 0.25 | 8, 16 | 0 at every width |
| `fingerprint_distribution_entropy` | 4 | 0.05 | 0.5 | 8, 16 | 0 at every width |
| `fingerprint_distribution_entropy` | 4 | 0.05 | 1 | 8, 16 | 0 at every width |

The sweep therefore shows no block-skipping benefit for this run; changing the selection method, n-gram size, frequency thresholds, or fingerprint width never changes the outcome.
