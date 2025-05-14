
- [Acknowledgements and Project Purpose](#acknowledgements-and-project-purpose)
- [Marimo vs. Jupyter: Project Focus](#marimo-vs-jupyter-project-focus)
  - [Initial impressions](#initial-impressions)
    - [Quarto integration](#quarto-integration)
  - [Jupyter vs. Marimo](#jupyter-vs-marimo)
  - [Tracking My Experience](#tracking-my-experience)
- [References](#references)


## Acknowledgements and Project Purpose

This repository is **forked from [cantaro86/Financial-Models-Numerical-Methods](https://github.com/cantaro86/Financial-Models-Numerical-Methods)**, a project by [Nicola Cantarutti](https://github.com/cantaro86) and contributors. Their extensive and well-documented work on quantitative finance models forms the foundation for this repo-many thanks to the original authors for making their research and code available under open source terms.

See the original [README](README-original.md) for the full project description.

---

## Marimo vs. Jupyter: Project Focus

This fork is being used as a **test case for evaluating Marimo** ([marimo.io](https://marimo.io)), a next-generation Python notebook environment, as an alternative to Jupyter notebooks. The primary goals are:

- **Porting and running selected notebooks** (starting with “1.1 Black-Scholes numerical methods”) in Marimo.
- **Documenting the migration process** and any usability differences.
- **Comparing features, workflows, and limitations** of Marimo and Jupyter for quantitative finance and data science use cases.
- **Collecting and sharing pros and cons** as discovered through practical use.

### Initial impressions

Marimo and Jupyter serve somewhat different audiences and use cases. Jupyter is deeply embedded in data science and education, with a huge ecosystem and broad tool support. Marimo, by contrast, is designed for reproducibility, interactivity, and maintainability, with features like:

- Notebooks as pure Python `.py` files (easy to version and execute as scripts)[6][8].
- Built-in UI elements and reactive execution, eliminating hidden state[6][8].
- Git-friendly, reproducible, and easily shareable as web apps[5][6][7].
- Automatic dependency tracking and sandboxed execution.

#### Quarto integration

https://marimo-team.github.io/quarto-marimo/

### Jupyter vs. Marimo

For a detailed technical comparison, see:

- [Jupyter vs Marimo: side-by-side (Deepnote)](https://deepnote.com/compare/jupyter-vs-marimo) [1]
- [JupyterLab vs Marimo: side-by-side (Deepnote)](https://deepnote.com/compare/jupyterlab-vs-marimo) [2]
- [Marimo FAQ: How is Marimo different from Jupyter?](https://docs.marimo.io/faq/) [6]
- [Marimo migration guide for Jupyter users](https://docs.marimo.io/guides/coming_from/jupyter/) [8]
- [EuroPython 2025: Meet Marimo, the next-gen Notebook](https://ep2025.europython.eu/session/meet-marimo-the-next-gen-notebook/)[5]
- [Reddit: Community discussion on Marimo vs Jupyter](https://www.reddit.com/r/Python/comments/1dvs2d6/reactive_notebook_for_python_an_alternative_to/)[7]
- [YouTube: Marimo vs Jupyter walkthrough](https://www.youtube.com/watch?v=tLyjRfkyfFg)[3]
- [LinkedIn: Reuven Lerner’s video on Marimo vs Jupyter](https://www.linkedin.com/posts/reuven_python-notebooks-marimo-vs-jupyter-activity-7318306191527362560-KUxu)[9]

---

### Tracking My Experience

As I progress through the migration and evaluation, I will:

- Update this README (or a dedicated document) with findings, pain points, and advantages discovered.
- Share code examples and workflow notes for both Marimo and Jupyter.
- Highlight any unique Marimo features or limitations relevant to financial modeling and reproducible research.

**If you have thoughts, suggestions, or experience with Marimo or Jupyter in quantitative finance, feel free to open an issue or discussion!**

---

*Original repo: [cantaro86/Financial-Models-Numerical-Methods](https://github.com/cantaro86/Financial-Models-Numerical-Methods)*
*Marimo: [marimo.io](https://marimo.io)*

---

## References

- [1]: https://deepnote.com/compare/jupyter-vs-marimo  
- [2]: https://deepnote.com/compare/jupyterlab-vs-marimo  
- [3]: https://www.youtube.com/watch?v=tLyjRfkyfFg  
- [4]: https://discourse.jupyter.org/t/jupyter-vs-marimo/28422  
- [5]: https://ep2025.europython.eu/session/meet-marimo-the-next-gen-notebook/  
- [6]: https://docs.marimo.io/faq/  
- [7]: https://www.reddit.com/r/Python/comments/1dvs2d6/reactive_notebook_for_python_an_alternative_to/  
- [8]: https://docs.marimo.io/guides/coming_from/jupyter/  
- [9]: https://www.linkedin.com/posts/reuven_python-notebooks-marimo-vs-jupyter-activity-7318306191527362560-KUxu

