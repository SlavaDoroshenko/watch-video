# watch-video

Claude Code skill: makes screen recordings (bug videos from testers) analysable.

- `scripts/video_bug.py analyze VIDEO` - metadata, per-frame scroll motion (phase correlation),
  jank events (stall / hitch / timing-gap / jump / reversal), overview contact sheets, burst sheets
  around events, motion plot, optional whisper transcript, `report.md`.
- `scripts/video_bug.py window VIDEO START END [--fps N] [--sheet]` - dump frames of a range.
- `scripts/video_bug.py rows VIDEO START END` - print motion.csv rows for a range.
- `scripts/video_bug.py probe VIDEO` - metadata JSON.
- `scripts/gen_test.py` - generates `test_cfr.mp4` with known defects for self-testing.

Requirements: ffmpeg/ffprobe, python3, numpy, pillow. Optional: whisper / whisper-cpp for narration.
Output goes to `~/.cache/claude-video/<video-name>/`.

Usage in Claude Code: `/watch-video ~/Downloads/bug.mp4 скролл дёргается на списке транзакций`
or simply mention the video path in a message.
