# Publishing checklist

- [x] Repository visibility confirmed as **public**.
- [ ] Choose a license if the repository or model will later be redistributed; none is selected automatically.
- [x] Do not commit videos, labels, private paths, H5 files, review renders, or checkpoints.
- [x] Publish snapshot-best-120.pt as the `v1.0.0` GitHub Release asset after explicit owner authorization.
- [x] Run the unit tests and `scripts/verify_release.py`.
- [x] Review every prospective tracked file before commit.
- [x] Include the model limitations with the checkpoint documentation.
