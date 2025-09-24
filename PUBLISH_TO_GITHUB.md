# How to publish this folder to GitHub

```powershell
cd "<PATH-TO-FOLDER>\github_ready"

# initialize
git init
git checkout -b main
git add .
git commit -m "Initial commit (GitHub-ready structure)"

# optional: enable Git LFS locally before first push
git lfs install

# create an empty repo on GitHub named: Thesis--Jump-risk-hedging-on-BTC-option
git remote add origin https://github.com/PhilPasu/Thesis--Jump-risk-hedging-on-BTC-option.git
git push -u origin main
```

## Tips
- If the remote already has a README, do: `git pull --rebase origin main` then `git push`.
- If large-file push fails, ensure Git LFS is installed and re-run the push.
