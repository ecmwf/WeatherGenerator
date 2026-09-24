# mh/port/flow-matching — conflict record

Branched off `mh/port/omit-target-prs` (task A). Replays the flow-matching commits from
`origin/flow-matching-pipeline` (tip `b0e82740`; `origin/mh/flow-matching-diagnostics-endpoint`
is the same commit). **All 7 applied.**

| new sha | original | date | subject |
|---|---|---|---|
| `ecec9a9f` | `2e5c344d` | 07-10 | init commit with flow and score matching |
| `e32d4eda` | `0d9e854c` | 07-13 | implemented loss tracking for x_0 mse |
| `d8b1e330` | `8958214b` | 07-13 | update denoiser, add sampling diagnostics |
| `2486ccb7` | `15081648` | 07-14 | reviewed code, added comments |
| `10e30020` | `503a1523` | 07-16 | embedded edm in flow-matching |
| `a57bae59` | `dca7e22a` | 07-20 | fixed sampling diagnostics and eval config  **(1 conflict, resolved)** |
| `c1c67813` | `b0e82740` | 07-22 | Plot terminal sampling node in flow-matching diagnostics |

## The one conflict — `src/weathergen/model/diffusion.py`, inference-step count

Both sides read the ODE denoising-step count from config, but disagreed on key, precedence and
default:

| | HEAD (develop + task A) | `dca7e22a` |
|---|---|---|
| `forward()` signature | `num_steps: int = 10` | `num_steps: int \| None = None` |
| body | `cf.get("fe_diffusion_num_inference_steps") or num_steps` | `if None: cf.get("fe_diffusion_num_steps", 10)` |
| key set in configs | 3 | **0** |
| effective value today | 10 | 10 |

### Findings

- **The `num_steps` parameter on `forward()` was dead.** Nothing in `src/`, `packages/` or
  `tests/` passes it — `model.py:894/899` call the engine without it. It existed only to hold
  the default `10`, which is why both sides needed a `config or param` dance.
- Consequently the precedence difference (arg-wins vs config-wins) could never fire.
- `fe_diffusion_num_inference_steps` is the live key: declared in
  `config/config_diffusion_d2048_forecast.yml`, `config/diffusion/config_diffusion_decoder.yml`
  and `config/diffusion/config_diffusion_forecast.yml`. `fe_diffusion_num_steps` is set by no
  config in the tree, including the ones these commits add.
- The two halves were not independently mixable: FM's `None` default with HEAD's `or` body
  yields `None` and breaks the sampler.

### Resolution (agreed with the author, not chosen unilaterally)

One knob, read from config, default 10, named `fe_diffusion_num_inference_steps`; the dead
parameter removed:

```python
def forward(
    self,
    tokens: torch.Tensor = None,
    fstep: int = None,
    meta_info: dict[str, SampleMetaData] = None,
    coords: torch.Tensor = None,
) -> torch.Tensor:
    ...
    # Number of ODE denoising steps, read from config. 10 is the historical
    # hardcoded value and remains the default when the key is unset.
    num_steps = self.cf.get("fe_diffusion_num_inference_steps", 10)
    return self.inference_forward(fstep=fstep, num_steps=num_steps, ...)
```

`num_steps` remains a parameter of `inference_forward`, `_run_ode`, `_stochastic_churn` and
`_plot_sampling_diagnostics` — those are genuinely passed and are untouched.

Behaviour is unchanged: every current config yielded 10 before and yields 10 now. What changes
is that `fe_diffusion_num_steps` is no longer consulted — nothing set it.

### Known inconsistency, deliberately left alone

`FlowMatching` (`flow_matching.py:252,289`) resolves its own step count a third way —
`cf.get("fm_num_steps", 50)` in `__init__`, applied as `num_steps or self.num_steps_default`.
Aligning it was considered and rejected: it would widen this PR into FM code these commits are
meant to introduce unmodified.
