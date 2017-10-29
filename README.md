# Not So Close Encounters

Contains the notebook and plot generator for my "Not So Close Encounters" blog post.

## Quickstart

To get started use the Makefile.

```shell
make create_environment

source activate not_so_close_encounters

make data # Downloads auxiliary datasets.

jupyter notebook
```

One thing to note is that `create_environment` _might_ not 100% work.
Installing geopandas is something of an ordeal in some cases because there are a lot of dependencies that require linking to C libraries which may or may not be version compatible.
I was able to get it to work with varying combinations of installing / uninstalling different dependencies, but honestly I couldn't say what the right procedure actually is.
My recommendation is to get a stiff drink and google through it.

## Makefile Targets

Ignoring the file-level targets:

| target                     | description                                                                                                                                       |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `create_fresh_environment` | Creates a blank conda environment named `not_so_close_encounters`.                                                                                |
| `create_environment`       | Creates a conda environment from `environment.yaml`.                                                                                              |
| `export_environment`       | Exports the environment to `environment.yaml`.                                                                                                    |
| `destroy_environment`      | Destroys the `not_so_close_encounters` environment.                                                                                               |
| `data`                     | Downloads and processes external datasets (The main dataset is accessed via the [data.world SDK](https://github.com/datadotworld/data.world-py)). |
| `plots`                    | Constructs only the plots from the notebook.                                                                                                      |

