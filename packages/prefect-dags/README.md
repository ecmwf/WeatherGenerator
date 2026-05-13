
# perfect-flows

The `perfect-flows` package lets you orchestrate  commands and SLURM jobs on various High Performance Computers (HPCs), abstracting away all the details of communicating with the various HPCs. It integrates with a modern beautiful UI and an industrial-grade scheduler on top of your HPC.

It solves the following problems:

1. Each HPC has different, incompatible interfaces to log in and submit jobs and commands.
2. The common denominator between most HPCs is low level (Linux environment and the Slurm scheduler).
3. The schedulers on deployed on HPCs are limited (no UI, limited job chaining) and complex to use correctly for long training jobs.
4. A standard connection to an HPC is usually valid for a hours or a few days, limiting the ability to run long-complex workflows

The perfect-flows package provides HPC-aware integration into the [Prefect](prefect.io) scheduler:

1. Provide a uniform way to run commands and jobs on various HPCs
2. It integrates into Prefect's tasks and flows to submit commands and Slurm jobs
3. The ability to interrupt, restart and modify workflows to minimize computation waste

## Quick start

This quick start runs the scheduler on your machine. It assumes you have access the following on your local machine:
- the [uv package manager](https://docs.astral.sh/uv/) version 0.9+
- an existing SSH connection to an HPC, for example `ssh hpc-login`, that does not require any further authentication (i.e. the command above should just log you into the HPC without prompting you for passwords, OTP codes, etc.)

1. Launch the prefect server:

```sh
uvx --with "prefect==3.7.0" prefect server start
```


2. Execute this script, which simply prints "hello world" through a Slurm job:

```python
from weathergen.prefect_dags.cmd_runners import EcmwfSshContext
from weathergen.prefect_dags import flow, run, sbatch

# The run context defines where the commands will be executed. 
ctx = EcmwfSshContext(host="hpc-login")

@flow(log_prints=True)
def test_run_cmd_flow(
    rerun_token=None,
):
    command = "echo 'hello world'"
    # Run a command on the HPC:
    cmd_result = run(ctx, command=command)
    print(f"Command result: {cmd_result.stdout.strip()}")
    slurm_result = sbatch(
        ctx,
        job_name="hello_world_job",
        command=command,
        time_limit="00:01:00",
    )
    print(f"Slurm job finished: {slurm_result.status}")
```

The whole file is on Github:
https://github.com/ecmwf/WeatherGenerator/blob/tjh/dev/prefect-test/packages/prefect-dags/examples/hello_world.py

You can run it from the terminal:



## Supported HPCs

Here is a summary table of the supported connections. All HPCs allow reusing the same connection without having to reauthenticate for some limited time (typically a day or two). After that, `perfect-flows` needs to be provided with refreshed keys or token. 
Some HPCs also provide an API-based interface that can be requested by users without asking for special privileges. This is the recommended route if you want to run workflows for more than a few days. 

This table summarises the supported connections:

| Center                                   | Session duration                     | Notes                                              |
| ---------------------------------------- | ------------------------------------ | -------------------------------------------------- |
| ECMWF<br>(HPC2020)                       | 2 days (ssh)<br>7 days (ecaccess)    | ecaccess can take a few minutes to launch commands |
| CSCS<br>- santis<br>- clariden<br>- alps | 1 day (ssh)<br>unlimited (Firecrest) |                                                    |
| JSC<br>- jupiter<br>- jureca<br>- juwels | 1 day (ssh)<br>6 months (UNICORE)    |                                                    |
| Cineca<br>- Leonardo                     | 2 days (ssh)                         |                                                    |
| BSC<br>- MareNostrum5                    | 2 days (ssh)                         | No official mechanism to extend a session          |
|                                          |                                      |                                                    |
