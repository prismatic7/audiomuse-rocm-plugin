# Agent instructions for audiomuse-rocm-plugin

## Fork-safety of GPU init

`register()` runs once in the worker's parent process, which then forks a child
per job. A HIP context does not survive `fork()`: any plugin code that
initializes HIP at worker start (e.g. `torch.cuda.is_available()`) poisons every
forked child — MIGraphX then dies on its first compile with "no kernel image is
available for execution on the device". GPU probes must run in the job child,
never in `register()`. The arch probe shells out to `rocminfo` for this reason;
details and the failure modes live in `docs/ARCH_NOTES.md`.

## Comments

- **Never comment on absence.** Don't write a comment explaining that
  something *isn't* there, *used to be* there, or *isn't being done* --
  "no libomp-dev here", "removed the X workaround", "we don't do Y
  anymore". A reader sees only the code that exists; a note about code
  that doesn't exist is unverifiable noise the moment they check. If a
  line was dropped, dropping it needs no comment -- the absence speaks
  for itself. Only comment to justify something *present*.
- **Comments say WHY, never WHAT.** Code should be legible enough that
  restating the WHAT in prose is dead weight. Write a comment only when
  there's a non-obvious reason behind the code.
- **Comments are not commit history.** Don't write "changed X to Y",
  "added this for the Z fix" -- that's what `git log`/`git blame` are
  for. A comment should read cold, with no idea what the last edit was.
