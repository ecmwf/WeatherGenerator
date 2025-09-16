## Description

<!--
Provide a brief summary of the changes introduced in this pull request.
-->


## Issue Number

<!--
Link the Issue number this change addresses:
Closes #XYZ 
-->

Is this PR a draft? Mark it as draft.

## Checklist before asking for review

-   [ ] I have performed a self-review of my code
-   [ ] My changes comply with basic sanity checks:
      - I have fixed formatting issues with `./scripts/actions.sh lint`
      - I have run unit tests with `./scripts/actions.sh unit-test`
      - I have documented my code and added unit tests if relevant
-   [ ] I have tried my changes with data and code:
      - I have run the integration tests with `./scripts/actions.sh integration-test`
      - (bigger changes) I have run a full training: `launch-slurm.py --time 60`
      - (bigger changes and experiments) I have shared a hegdedoc in the github issue with all the configurations and runs for this experiments
-   [ ] I have informed and aligned with people impacted by my change:
    - for config changes: the MatterMost channels and/or a design doc
    - for changes of dependencies: the MatterMost software development channel
