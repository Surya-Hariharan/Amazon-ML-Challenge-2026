# resources/

Raw, unmodified materials provided by the Amazon ML Challenge 2026 organizers.

## `student_resource.zip`

The original organizer bundle (`6ab10eb3b23ba_student_resource.zip` as distributed, ~1.1
GB), renamed for clarity. **Do not extract a copy of its `dataset/` into the repo
working tree except into the gitignored `dataset/` directory at repo root** — see
[`../dataset/README.md`](../dataset/README.md) for the regeneration commands.

It contains (per `unzip -l resources/student_resource.zip`):

- `student_resource/dataset/train/` and `student_resource/dataset/test/` — the same
  train/test TSVs described in `../dataset/README.md`.
- `student_resource/utils/validate_submission.py` — identical to
  `../utils/validate_submission.py`.
- `student_resource/README.md` — the organizer's original problem-statement/rules
  document. Its content has been split and reproduced verbatim into
  [`../docs/challenge/problem_statement.md`](../docs/challenge/problem_statement.md),
  [`../docs/challenge/submission_requirements.md`](../docs/challenge/submission_requirements.md)
  and [`../docs/challenge/guidelines.md`](../docs/challenge/guidelines.md) for easier
  reference; read those three files rather than re-extracting the zip for day-to-day
  work.
- `student_resource/Documentation_template.md` — the blank official methodology
  template. This project's working copy to fill in lives at
  [`../docs/methodology/Documentation_template.md`](../docs/methodology/Documentation_template.md)
  (currently still a placeholder — see that file and `plan.md` CP12 for when/how it gets
  filled in). If that placeholder is ever lost, re-extract the real template from this
  zip:
  ```bash
  unzip -p resources/student_resource.zip student_resource/Documentation_template.md \
    > docs/methodology/Documentation_template.md
  ```

## Why keep the zip at all

It is the single authoritative, unmodified copy of what the organizers handed out —
useful for re-verifying the dataset, the validator script, or the original wording of
the rules if any question arises about what changed on our side versus what was given.
It is **not** part of the final submission zip (see
`../docs/challenge/submission_requirements.md`) and should stay out of it.

Two convenience copies that used to sit at the repo root —
`student_resource_README.md` and `student_resource_Documentation_template.md` — were
removed during the 2026-09-25 repo reorganisation after confirming (via `diff`) they
were byte-identical to `student_resource/README.md` and
`student_resource/Documentation_template.md` inside this zip; their content lives on
here and, for the README, in `../docs/challenge/`.
