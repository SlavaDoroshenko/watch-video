---
name: watch-video
description: >
  "Watch" a video file (screen recording from a tester: .mp4/.mov/.webm/.mkv/.gif) — turns it into
  timestamped contact sheets, a per-frame scroll-motion timeline and burst sheets around every
  hitch/freeze/reversal, so the bug can be seen and located in time. Use whenever the user gives a path
  to a video, mentions a screen recording / видео / запись экрана of a bug (scroll дёргается, лагает,
  фризит, мигает, анимация ломается), asks what happens in a video, or invokes /watch-video.
---

# Watch video

Claude Code cannot open video files directly, and a screenshot is not enough for motion bugs.
This skill converts the video into things you *can* read: images with timestamps and a numeric
motion timeline. Never tell the user a video cannot be analysed - run the pipeline.

If invoked as `/watch-video <path> [what to look for]`, the arguments are: $ARGUMENTS

Script: `~/.claude/skills/watch-video/scripts/video_bug.py` (needs `ffmpeg`, `python3` + numpy + pillow;
all present on this Mac; fallback `uv run --with numpy --with pillow video_bug.py ...`).

## Workflow

1. **Analyse** (about 3 s for 16 s @ 60 fps; long videos - pass `--start/--end`):
   ```bash
   python3 ~/.claude/skills/watch-video/scripts/video_bug.py analyze "<video>"
   ```
   Output dir: `~/.cache/claude-video/<video-name>/` (override with `--out`). Files:
   `report.md`, `summary.json`, `motion.png` (plot), `motion.csv` (per-frame table),
   `sheet_NN.jpg` (overview, ~2 fps), `event_NN_<type>_<time>.jpg` (frame bursts at native fps),
   `frames_overview/`, `frames_events/`, optional `audio.txt` transcript.
2. **Read `report.md`** - verdict, scroll gestures table, events table with timestamps.
3. **View `motion.png`** with the Read tool. Top panel = scroll delta per frame (+ = scrolling down,
   content moving up); bottom = frame interval. Blue bands = gestures, vertical lines = events.
4. **View every `sheet_NN.jpg` in order** - this is "watching" the video. Describe the UI flow
   (what screen, what the tester does, where the visible problem is).
5. **View the burst sheet of each event** (`event_NN_*.jpg`). Consecutive frames; red border marks the
   bad frames; label = `f<frame> t=<time> d=<scroll delta px> dt=<ms since previous frame>`.
6. **Zoom when needed:**
   ```bash
   # every frame of a range (fps 0 = native) as separate images, plus a grid with --sheet
   python3 ~/.claude/skills/watch-video/scripts/video_bug.py window "<video>" 12.2 12.6 --sheet
   # numeric rows for a range (frame, t, dt_ms, d_offset_px, dx_px, conf, mean_abs_diff, duplicate)
   python3 ~/.claude/skills/watch-video/scripts/video_bug.py rows "<video>" 12.2 12.6
   # different ROI / sub-range / finer pixels
   python3 ~/.claude/skills/watch-video/scripts/video_bug.py analyze "<video>" --start 10 --end 14 --width 540 --crop-top 0.18 --crop-bottom 0.15
   ```
7. **Correlate with the user's description**, then go to the code (see "Where to look").
   If several gestures could be "the bug", say which candidates you see with timestamps and why.

Useful flags: `--crop-top/--crop-bottom` (default 0.10 each: excludes status bar / tab bar; raise them when
a sticky header, bottom sheet or FAB sits inside the ROI), `--width` (360 default; 540 for finer px),
`--min-move`, `--max-events`, `--sheet-fps`, `--max-frames`, `--grid 5x2`, `--no-audio`, `--no-motion`, `--no-sheets`.

## How to read the motion data (jank signatures -> likely causes)

| Signature | Meaning | Usual suspects |
|---|---|---|
| **stall / hitch**: duplicate frames mid-fling, then a catch-up jump | Main thread missed frames: the scroll position kept advancing in the model but no frames were produced | Heavy row/cell creation (no reuse, images decoded on main thread, deep layout), synchronous work in scroll callbacks (`scrollViewDidScroll`, RN `onScroll` + `setState`, Flutter scroll listener + `setState`), AutoLayout thrash, shadow/blur/mask layers in cells, RN JS thread busy, Flutter shader compilation on first scroll, `ListView` without `itemExtent`, logging in scroll handlers |
| **timing-gap, content did NOT advance** | VFR recorder (Android `screenrecord`, emulators) got no frames and the scroll clock stalled: UI thread blocked | Same as above |
| **timing-gap, content caught up proportionally** | No frames for the gap, then the fling physics caught up: dropped frames while moving. Almost always app jank; a recorder-side skip is possible but rare | Same as above; if the user says it looked smooth on the device, suspect the recorder |
| **reversal, jump back** (1-2 frames, large) | Something wrote the scroll offset during the user's scroll, or content above the viewport changed height | `setContentOffset`/`scrollTo`/`jumpTo` from a state update; items prepended without `maintainVisibleContentPosition`; row height changing after async image/data load (estimated vs actual size); keyboard-avoidance changing insets; `contentInset` toggled; nested scrolls fighting (carousel inside a list, `nestedScrollEnabled`, `bounces`); pull-to-refresh; `scrollEnabled` flipped |
| **reversal, drift back** (4+ frames, small steps, ~100-300 ms) | An animation moves the content against the user's direction after/while they scroll: snap, bounce or an offset correction | Collapsing/expanding header with snap logic (`snapToInterval`, `pagingEnabled`, custom snap on `scrollEndDrag`), `scrollTo` with animation from a state change, `contentInset` animation, `UIScrollView` bounce at an edge (normal), `KeyboardAvoidingView` |
| **reversal only at gesture end** | Rubber-band bounce at the content edge | Normal on iOS unless huge, slow, or the content was not at an edge |
| **two-state LAYOUT FLICKER** (screen alternates between exactly two layouts; identical +/-N px steps and identical frame diffs; happens even without a scroll) | An element ~N px tall is inserted/removed or shown/hidden every few frames, pushing everything below it. Looks like "the screen twitches" | A conditional render flapping (`{cond && <Banner/>}` where `cond` flips on every render), state derived from a layout measurement (`onLayout`/`GeometryReader`/`LayoutBuilder` -> `setState` -> re-layout -> different measurement), two effects/subscriptions overwriting the same flag, placeholder/skeleton swapping with real content, `Animated.loop`/`repeat` on height or opacity, `LayoutAnimation` re-triggered on each render, keyboard show/hide events toggling a view, SwiftUI `if` on state mutated in `onAppear`/`onChange` of that same state, Flutter `setState` inside `build`. Measure the step size: it equals the height of the culprit element |
| **oscillation** (3+ sign flips within a few frames, uneven steps) | Feedback loop: scroll -> layout/offset change -> scroll correction | Header collapse based on offset thresholds; `onLayout`-driven sizing; two scroll views syncing each other; `GeometryReader`/`PreferenceKey` inside rows; `contentInset` set in `scrollViewDidScroll` |
| **0 / 2x alternating pattern through a whole gesture** | The app renders every other frame (30 fps on a 60 Hz display) | Scroll driven from a 30 fps source (RN `Animated` without `useNativeDriver`, JS-driven scroll), per-frame work just above 16 ms, Low Power Mode, `preferredFramesPerSecond`, ProMotion mismatch |
| **isolated jump without a preceding stall** | One abnormally large step | Programmatic scroll (`scrollToIndex`, `scrollTo`), content inserted above, or just finger acceleration (benign) |
| **verdict SMOOTH but the user says "дёргается"** | The artefact is not scroll offset: flicker, images popping in, text reflow, blank cells, layout shift in place | Look for frames with `mean_abs_diff` high while `d_offset_px` = 0 (`rows`), then `window` there. Cause: re-render/flicker, placeholder swap, recycled cell showing stale content |
| **scene-change** markers | Navigation, modal, big UI change | Context only |

Flicker spans are detected independently of scroll gestures and are excluded from gesture segmentation, so the
verdict starts with `LAYOUT FLICKER DETECTED at <spans>` when present; scroll findings follow after it.

Recorder quirks: iOS/macOS recordings are constant fps and *repeat* frames when the app stalls (=> stalls).
Android/emulator recordings are variable fps and *omit* frames (=> timing-gaps). The analyser handles both.
A stall while the finger is down may also be the finger pausing - check the burst sheet; a stall during
a fling (finger lifted) is jank for sure.

## Where to look in the code (grep starters)

- **iOS / UIKit**: `setContentOffset`, `scrollViewDidScroll`, `contentInset`, `reloadData`/`performBatchUpdates` during scroll, `invalidateLayout`, `layoutIfNeeded` in cell config, image decoding on main thread, `estimatedRowHeight`.
- **SwiftUI**: `ScrollViewReader`/`scrollTo`, `.onChange` mutating state the ScrollView depends on, `LazyVStack` rows with variable height, `GeometryReader`/`PreferenceKey` in rows, `.id()` changes forcing re-creation.
- **React Native**: `onScroll` handlers with `setState`, `scrollEventThrottle`, `Animated.event` without `useNativeDriver`, `FlatList` `getItemLayout`/`keyExtractor`/`initialNumToRender`, `maintainVisibleContentPosition`, `KeyboardAvoidingView`, nested `ScrollView`/`FlatList`, Reanimated worklets doing layout, `onLayout` loops.
- **Flutter**: `setState` in a scroll listener / `NotificationListener`, `ListView` without `itemExtent`/`prototypeItem`, `shrinkWrap: true` inside a scrollable, `Opacity`/`ClipRRect`/`BackdropFilter`/shadows in items, `controller.jumpTo/animateTo`, `Sliver` size changes, image decoding size.
- **Android**: `notifyDataSetChanged`, `scrollToPosition`/`smoothScrollBy`, `NestedScrollView` + `RecyclerView`, `requestLayout` in `onBind`, `CoordinatorLayout` behaviours, `setHasFixedSize`.
- **Web**: `scrollTop =`, `scrollIntoView`, layout thrash in scroll listeners (read `offsetHeight` then write), `position: sticky` + transforms, `overflow-anchor`, images without dimensions, `scroll-behavior: smooth`.

## Reporting to the user

- Answer in the user's language. Give timestamps as `m:ss.mmm` and say what is visible there.
- Separate what is *measured* (stall 200 ms at 12.267, jump 56 px) from what is *inferred* (probable cause).
- Name the top 2-3 code suspects and what to grep/check, then go check them if the project is open.
- Token budget: one sheet is roughly 2-5k tokens. Read sheets and event bursts, not the raw `frames_*` files, unless a specific frame is needed.
- Audio: transcribed only if a whisper CLI is installed (`pip3 install openai-whisper` or `brew install whisper-cpp`); otherwise the report says whether narration exists.

## Testing the tool itself

Synthetic reference clips with known defects (reversal at 1.500 s, 10-frame freeze at 2.500 s, catch-up jump, smooth reverse scroll) can be regenerated with `scripts/gen_test.py`; the analyser must flag exactly those.
