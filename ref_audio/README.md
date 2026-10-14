# Reference audio: trusted local files only

Keep only reference audio for which you have appropriate consent, rights and
provenance. The application looks for `<voice>.wav`, `<voice>.mp3`, `<voice>.flac`,
`<voice>.aac`, `<voice>.ogg`, or `<voice>.pcm` under this directory.

Use real, singly linked regular files under a non-symlinked directory tree.
Symlinked and multiply linked reference inputs are rejected. The optional
processed-audio cache defaults to `ref_audio_cache/`; keep that directory
private to trusted writers as well. An unexpected pre-existing
`<voice>_processed.pcm` stops conversion rather than being overwritten.

This is a pathname-metadata guard, **not** a protection against concurrent
filesystem swaps, a guarantee of codec/file content, or a rights audit.
Do not expose either directory to less-privileged writers.
See the root README for limitations and mock-only verification status.
