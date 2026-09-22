# GraphVSelect — project page

Source of the project page for **"GraphVSelect: Verified Graph-guided Selection for Zero-shot
Referring Expression Comprehension"**. This branch (`webpage`) is served by GitHub Pages; the
code lives on `main`.

- `index.html` — the page (single static file, no build step)
- `static/css/`, `static/js/` — Bulma and Font Awesome from the
  [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template)
- `static/images/` — paper figures rendered from the submission PDFs

To preview locally:

```bash
python3 -m http.server 8000
# then open http://localhost:8000/
```

The page is licensed under [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/);
the code on `main` is under Apache-2.0.
