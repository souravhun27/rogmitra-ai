# Deployment guide: RogMitra.AI

The deployed app **loads** a trained model; it never trains. So the repository you deploy must already contain the
trained files. Follow the checklist, then pick one option.

## 0. Pre-flight checklist (do this once)

1. Train and evaluate on your own machine: `bash run_all.sh` (or the individual commands in the README).
2. Open each link in `data/remedies.json`, check the summary against the page, and set `"verified": true` for entries you checked.
   `python -m src.check_remedies --online` tests that each link opens (it does not replace reading the page).
3. Run `python -m src.preflight`. It must end with *All required files are present.* It also warns if `.gitignore`
   would stop a needed file from being uploaded.
4. Run `python -m pytest -q`. All tests should pass.
5. Take your own screenshots into `docs/screenshots/` and run `python -m src.update_readme` so the README shows your numbers.

What must be in the repository: `app/`, `src/`, `models/` (small: `mlp.pt`, `symptoms.json`, `classes.json`),
`data/processed/`, `data/remedies.json`, `reports/`, `requirements.txt`, `requirements-deploy.txt`, `app/requirements.txt`,
`Dockerfile`, `.streamlit/config.toml`. What must **not** be in it: `kaggle.json`, `.env`, `.venv/`, and the raw Kaggle
CSV files (`data/raw/` is git-ignored).

## 1. Put the project on GitHub

```bash
git init
git add .
git status                       # check that kaggle.json and .venv are NOT listed
git commit -m "RogMitra.AI: explainable disease prediction"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

## 2. Option A: Streamlit Community Cloud (easiest)

1. Sign in at <https://share.streamlit.io> with GitHub and choose **Create app**.
2. Pick your repository and branch (`main`). Set **Main file path** to `app/Home.py`.
3. Under **Advanced settings**, choose Python 3.11 (any of 3.10 to 3.12 matches the pinned packages).
4. Deploy. Streamlit looks for a dependency file first in the folder of the main file, then in the repository root,
   so it installs the lean list in `app/requirements.txt` (CPU PyTorch, no training-only packages).

Notes
* Free apps have limited memory (roughly 690 MB guaranteed, up to about 2.7 GB according to Streamlit's documentation) and
  go to sleep when idle; the first visit after a sleep takes longer to start.
* If the log says the app exceeded its resource limits, make sure `models/text_model/` is not in the repository and use
  Option B.
* After changing `requirements` files, reboot the app from its dashboard.

## 3. Option B: Hugging Face Spaces (Docker)

1. Create a new Space at <https://huggingface.co/new-space>: choose **Docker** as the SDK, hardware **CPU basic** (free).
2. In the repository that you push to the Space, the top of `README.md` must start with this metadata block (add it to the
   Space copy; GitHub will show it as a small table, which is fine):

   ```yaml
   ---
   title: RogMitra.AI
   emoji: 🩺
   colorFrom: orange
   colorTo: red
   sdk: docker
   app_port: 7860
   pinned: false
   ---
   ```
3. Push the code:

   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   git push space main
   ```
   (Hugging Face asks for your username and an access token with write permission, created under *Settings, Access Tokens*.)
4. The Space builds the `Dockerfile` and starts the app on port 7860, which the `Dockerfile` already uses.

## 4. Option C: Docker on any machine

```bash
docker build -t rogmitra-ai .
docker run --rm -p 8501:7860 rogmitra-ai
# open http://localhost:8501
```

The image uses CPU-only PyTorch and the lean `requirements-deploy.txt`. Training is not part of the image; train first, then build.

> The Dockerfile was written and its dependency list was checked by importing the app with the training-only packages blocked,
> but the image itself was **not built** while this guide was written. If the build fails, read the first red error line:
> it is almost always a package version or the missing `models/` folder.

## 5. Check the deployed app on a phone

Open the public link on your own phone and check: the sidebar arrow (top left) opens the page menu and the Light/Dark
buttons; the top-3 bars and the contribution chart fit the screen; tables scroll sideways; **Predict**, a follow-up
suggestion and **Download PDF** work.

## Troubleshooting

| Problem | Likely cause and fix |
|---|---|
| `FileNotFoundError: models/mlp.pt` | `models/` was not committed. Check `git ls-files models` and `.gitignore`. |
| Model Performance page empty | `reports/` was not committed, or `evaluate` was never run. |
| "ImportError: pyarrow requires NumPy 2.0" | Package versions drifted. Use the pinned `requirements-deploy.txt`. |
| Build is very slow or out of memory | The default CUDA build of PyTorch was installed. Keep the `+cpu` line and `--extra-index-url` in `requirements-deploy.txt`. |
| App sleeps / first load slow | Normal on free hosting. Open it once before your demo. |
