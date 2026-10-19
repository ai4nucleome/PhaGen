# PhaGen project page

Static website prepared for `https://ai4nucleome.github.io/PhaGen/`.
Page files are in `public/`; no build or dependency installation is required.

Preview from this directory:

```bash
python -m http.server 18765 --bind 127.0.0.1 --directory public
```

After uploading the repository, select **Settings → Pages → Source → GitHub
Actions**, then **Actions → Deploy PhaGen project page → Run workflow** for the
first deployment. Later page changes on `main` deploy automatically. The workflow
in `.github/workflows/deploy-pages.yml` publishes only `website/public/`.

The page supports English and Chinese, with an overview, paper Figure 1, seven
prediction benchmarks, installation instructions, and resource links. The landscape
is an original AI-generated illustration. The Zenodo link will be added when the
archive is released.

A generation animation compares PhaGen's masked diffusion with Evo's left-to-right
autoregressive decoding. It uses a shared prefix and illustrative sequence; playback
timing and iteration counts are not measured model performance.
