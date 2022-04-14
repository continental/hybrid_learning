Scripts
=======

This folder holds scripts for application of functionalities and utility scripts.

Execution
---------

All scripts must be run using `python 3.8` and must be called from the project
root.
The scripts use the experiment framework [sacred](https://sacred.readthedocs.io/en/stable/index.html),
so all provided `exp_*.py` scripts can be called according to the sacred
[command-line interface](https://sacred.readthedocs.io/en/stable/command_line.html).  
For example, to list available configuration values use

```bash
python -m script.exp_ca.exp_<ca_script_spec>.py print_config  # for concept analysis experiments
python script/exp_<script_spec>.py print_config  # for other experiments
```
and for all available named configs
```bash
python -m script.exp_ca.exp_<ca_script_spec>.py print_named_configs  # for concept analysis experiments
python script.exp_<script_spec>.py print_named_configs  # for other experiments
```
Some other useful options (for more see the sacred docs):

- `-F <log_root>`: enable file logging of the experiment
- `--pdb`: in case of any failure, directly python debugger
- `-l INFO`: set log-level to `INFO` (i.e. log the current analysis status) 


Documentation
-------------
For documentation about the scripts and the used configs have a look at the
docstrings of the scripts.
For documentation about the config options:

- for an overview: Use the sacred config list commands for an overview of available configuration options.
- for details on an option: Have a look the comments in the functions decorated with `@ex.config`
  in the experiment script files respectively the `script/exp_ca/config/*.py` files.

