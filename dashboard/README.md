# Streamlit Dashboard

This dashboard now follows a modular tab architecture:

- Each concept lives in its own Python file under `dashboard/tabs/`
- The Streamlit app imports those modules through `dashboard/tab_registry.py`
- Tabs render analysis directly from code instead of running notebook cells or loading saved image folders

## Current Tabs

- `COVID Sector Analysis`: self-contained multifractal COVID analysis with seven plots rendered directly in Streamlit

## Add A New Tab

1. Create a new module under `dashboard/tabs/`
2. Export a `TAB_SPEC`
3. Register it in `dashboard/tab_registry.py`

## Run

From the repository root:

```bash
streamlit run dashboard/app.py
```
