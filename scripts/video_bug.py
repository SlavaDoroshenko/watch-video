#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video_bug.py - make a screen recording readable for Claude Code.

  analyze  VIDEO            full pipeline: probe + motion analysis + contact sheets
                            + event bursts + (optional) audio transcript + report.md
  window   VIDEO START END  dump individual frames for a time window (zoom-in)
  probe    VIDEO            metadata only (JSON)

Motion analysis estimates, for every pair of consecutive frames, how far the
screen content moved (phase correlation on a grayscale, downscaled ROI) and
flags scroll jank: stalls (duplicate frames), timing gaps (VFR recorders),
jumps (catch-up after a dropped frame) and reversals (content jumps backwards).

Requires ffmpeg + ffprobe on PATH, python3 with numpy and pillow.
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
except ImportError as e:  # pragma: no cover
    sys.exit(
        f"missing python dependency: {e}\n"
        "install:  pip3 install numpy pillow\n"
        "or run:   uv run --with numpy --with pillow video_bug.py ..."
    )

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".gif", ".mpg", ".mpeg", ".3gp"}
CONF_MIN = 0.03        # phase-correlation peak below this = unreliable match
DUP_MAD = 0.6          # mean abs diff (0-255) below this = duplicate frame
FLAT_STD = 2.0         # ROI with std below this has no texture to track


# --------------------------------------------------------------------------- utils
def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def need(tool):
    if not shutil.which(tool):
        sys.exit(f"'{tool}' not found on PATH. Install with: brew install ffmpeg")


def fmt_t(t):
    if t is None or (isinstance(t, float) and math.isnan(t)):
        return "?"
    m = int(t // 60)
    s = t - m * 60
    return f"{m}:{s:06.3f}"


def even(x):
    x = int(round(x))
    return x if x % 2 == 0 else x + 1


def ffmpeg_version():
    r = run(["ffmpeg", "-version"])
    m = re.search(r"ffmpeg version (\d+)\.(\d+)", r.stdout)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def passthrough_flags():
    return ["-fps_mode", "passthrough"] if ffmpeg_version() >= (5, 1) else ["-vsync", "0"]


_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


def font(size=14):
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # old pillow
        return ImageFont.load_default()


def parse_showinfo_pts(stderr_text):
    return [float(x) for x in re.findall(r"pts_time:\s*(-?[0-9.]+)", stderr_text)]


# --------------------------------------------------------------------------- probe
def probe(video):
    need("ffprobe")
    r = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", video])
    if r.returncode:
        sys.exit(f"ffprobe failed: {r.stderr.strip()}")
    info = json.loads(r.stdout)
    v = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    a = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)
    if not v:
        sys.exit("no video stream found")

    def frac(x):
        try:
            n, d = x.split("/")
            return float(n) / float(d) if float(d) else 0.0
        except Exception:
            return 0.0

    rot = 0
    for sd in v.get("side_data_list") or []:
        if "rotation" in sd:
            rot = int(round(float(sd["rotation"])))
    if "rotate" in (v.get("tags") or {}):
        rot = int(v["tags"]["rotate"])
    w, h = int(v["width"]), int(v["height"])
    if rot % 180:
        w, h = h, w
    dur = float(info.get("format", {}).get("duration") or v.get("duration") or 0)
    return dict(
        path=str(video), size_mb=round(os.path.getsize(video) / 1e6, 1),
        width=w, height=h, rotation=rot, codec=v.get("codec_name"), pix_fmt=v.get("pix_fmt"),
        duration=dur, avg_fps=round(frac(v.get("avg_frame_rate", "0/1")), 3),
        r_fps=round(frac(v.get("r_frame_rate", "0/1")), 3),
        nb_frames=int(v.get("nb_frames") or 0), has_audio=a is not None,
        audio_codec=a.get("codec_name") if a else None,
    )


# --------------------------------------------------------------------------- frame streaming
def stream_gray(video, width, height, start, end):
    """Yield (pts_time, uint8 HxW) for every decoded frame, exact timestamps from showinfo."""
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-loglevel", "info"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", video]
    if end:
        cmd += ["-t", f"{max(0.0, end - (start or 0)):.3f}"]
    cmd += ["-an", "-sn", "-dn", "-vf", f"scale={width}:{height}:flags=area,format=gray,showinfo"]
    cmd += passthrough_flags() + ["-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    err_chunks = []

    def drain():
        err_chunks.append(proc.stderr.read())

    th = threading.Thread(target=drain, daemon=True)
    th.start()
    nbytes = width * height
    frames = []
    while True:
        buf = proc.stdout.read(nbytes)
        if len(buf) < nbytes:
            break
        frames.append(np.frombuffer(buf, dtype=np.uint8).reshape(height, width))
    proc.wait()
    th.join()
    stderr = b"".join(err_chunks).decode("utf-8", "replace")
    pts = parse_showinfo_pts(stderr)
    return frames, pts, stderr


def phase_shift(F_prev, cur_w, shape):
    """Return (shift_y, shift_x, peak). cur = prev shifted by (shift_y, shift_x)."""
    G = np.fft.rfft2(cur_w)
    R = np.conj(F_prev) * G
    R /= (np.abs(R) + 1e-9)
    r = np.fft.irfft2(R, s=shape)
    idx = int(np.argmax(r))
    py, px = divmod(idx, shape[1])
    if py > shape[0] // 2:
        py -= shape[0]
    if px > shape[1] // 2:
        px -= shape[1]
    return py, px, float(r.flat[idx]), G


# --------------------------------------------------------------------------- motion analysis
def analyze_motion(video, meta, out, args):
    scale = args.width / meta["width"]
    W = args.width
    H = even(meta["height"] * scale)
    frames, pts, stderr = stream_gray(video, W, H, args.start, args.end)
    n = len(frames)
    if n < 3:
        return None, f"only {n} frames decoded; motion analysis skipped\n{stderr[-500:]}"
    t0 = args.start or 0.0
    if len(pts) == n:
        t = np.array(pts, dtype=np.float64) + t0
    else:
        fps = meta["avg_fps"] or meta["r_fps"] or 30.0
        t = np.arange(n) / fps + t0
        print(f"[warn] showinfo gave {len(pts)} timestamps for {n} frames; using uniform {fps:.2f} fps", file=sys.stderr)

    y0 = int(H * args.crop_top)
    y1 = int(H * (1 - args.crop_bottom))
    roi_shape = (y1 - y0, W)
    win = (np.hanning(roi_shape[0])[:, None] * np.hanning(roi_shape[1])[None, :]).astype(np.float32)

    d_off = np.zeros(n)      # change of scroll offset (content px, original scale); + = scrolled down
    d_x = np.zeros(n)
    conf = np.zeros(n)
    mad = np.zeros(n)
    dup = np.zeros(n, dtype=bool)
    flat = np.zeros(n, dtype=bool)

    prev_roi = None
    F_prev = None
    for i, fr in enumerate(frames):
        roi = fr[y0:y1].astype(np.float32)
        if roi.std() < FLAT_STD:
            flat[i] = True
        cur_w = (roi - roi.mean()) * win
        if prev_roi is None:
            F_prev = np.fft.rfft2(cur_w)
            prev_roi = roi
            conf[i] = 1.0
            continue
        m = float(np.mean(np.abs(roi - prev_roi)))
        mad[i] = m
        if m < DUP_MAD:
            dup[i] = True
            conf[i] = 1.0
            G = np.fft.rfft2(cur_w)
        elif flat[i] or flat[i - 1]:
            conf[i] = 0.0
            d_off[i] = np.nan
            G = np.fft.rfft2(cur_w)
        else:
            sy, sx, peak, G = phase_shift(F_prev, cur_w, roi_shape)
            conf[i] = peak
            if peak < CONF_MIN:
                d_off[i] = np.nan
                d_x[i] = np.nan
            else:
                d_off[i] = -sy / scale   # content moved down by sy => offset decreased
                d_x[i] = -sx / scale
        F_prev = G
        prev_roi = roi
    del frames

    dt = np.diff(t, prepend=t[0])
    dt[0] = np.nan
    return dict(t=t, dt=dt, d=d_off, dx=d_x, conf=conf, mad=mad, dup=dup, flat=flat,
                n=n, scale=scale, W=W, H=H, roi=(y0, y1)), None


def detect_events(mo, min_move):
    t, dt, d, conf, mad, dup = mo["t"], mo["dt"], mo["d"], mo["conf"], mo["mad"], mo["dup"]
    n = mo["n"]
    med_dt = float(np.nanmedian(dt[1:])) if n > 1 else 1 / 30
    fps_eff = 1 / med_dt if med_dt > 0 else 30.0
    dd = np.nan_to_num(d, nan=0.0)
    moving = (np.abs(dd) >= min_move) & (conf >= CONF_MIN)
    gap = max(2, int(round(0.25 * fps_eff)))  # merge pauses up to 250 ms into a gesture
    events = []

    # ---- oscillation / two-state layout flicker: dense sign flips, independent of gestures
    mov_idx = [k for k in range(1, n) if moving[k]]
    flips = [mov_idx[m] for m in range(1, len(mov_idx))
             if np.sign(dd[mov_idx[m]]) != np.sign(dd[mov_idx[m - 1]])]
    flip_gap = max(3, int(round(0.25 * fps_eff)))
    osc_spans = []
    k0 = 0
    while k0 < len(flips):
        k1 = k0
        while k1 + 1 < len(flips) and flips[k1 + 1] - flips[k1] <= flip_gap:
            k1 += 1
        if k1 - k0 + 1 >= 3:
            prev_mov = max([m for m in mov_idx if m < flips[k0]], default=flips[k0])
            osc_spans.append((prev_mov, flips[k1], flips[k0]))
        k0 = k1 + 1
    osc_mask = np.zeros(n, dtype=bool)
    for (o0, o1, first_flip) in osc_spans:
        osc_mask[o0:o1 + 1] = True
        span_mov = [k for k in mov_idx if first_flip <= k <= o1]  # stats over the flapping part only
        amps = np.abs(dd[span_mov])
        mads = mad[span_mov]
        nfr = o1 - o0 + 1
        ms = float((t[o1] - t[o0]) * 1000)
        amp = float(np.median(amps))
        mad_med = float(np.median(mads)) if len(mads) else 0.0
        two_state = len(amps) >= 4 \
            and float(np.mean(np.abs(amps - amp) <= max(min_move, 0.1 * amp))) >= 0.8 \
            and float(np.mean(np.abs(mads - mad_med) <= 0.15 * max(1e-6, mad_med))) >= 0.8
        n_flips = sum(1 for f in flips if o0 <= f <= o1)
        if two_state:
            detail = (f"two-state LAYOUT FLICKER: the screen alternates between exactly two layouts {n_flips} times in "
                      f"{ms:.0f} ms, content shifted by ~{amp:.0f}px each time = an element ~{amp:.0f}px tall is inserted/removed "
                      f"repeatedly (conditional render / visibility toggling / animation loop). Not a scroll.")
        else:
            detail = (f"content shakes back and forth {n_flips} times in {ms:.0f} ms (~{amp:.0f}px per step): "
                      f"feedback loop (scroll -> layout/offset change -> scroll) or layout thrash")
        events.append(dict(frame=o0, t=float(t[o0]), type="oscillation", gesture=0, detail=detail, note="",
                           severity=3, frames=nfr, ms=ms, amp=amp, two_state=bool(two_state)))
    seg_moving = moving & ~osc_mask

    segs = []
    i = 0
    while i < n:
        if seg_moving[i]:
            j, last = i, i
            while j < n and (seg_moving[j] or j - last <= gap):
                if seg_moving[j]:
                    last = j
                j += 1
            segs.append((i, last))
            i = j
        else:
            i += 1
    segs = [(a, b) for a, b in segs
            if np.sum(seg_moving[a:b + 1]) >= 4 and abs(np.nansum(dd[a:b + 1])) >= max(20, 4 * min_move)]

    gestures = []
    for gi, (a, b) in enumerate(segs, 1):
        seg = dd[a:b + 1]
        mv = moving[a:b + 1]
        dom = 1 if np.sum(seg) >= 0 else -1
        med = float(np.median(np.abs(seg[mv]))) if mv.any() else min_move
        total = float(np.sum(seg))
        dur = float(t[b] - t[a]) if b > a else 0.0
        g = dict(id=gi, start=float(t[a]), end=float(t[b]), frames=b - a + 1, total_px=total,
                 direction="down (content moves up)" if dom > 0 else "up (content moves down)",
                 median_px_per_frame=med, px_per_s=(total / dur if dur > 0 else 0.0), events=[])
        gestures.append(g)
        length = b - a + 1

        def against(k):
            return conf[k] >= CONF_MIN and moving[k] and np.sign(dd[k]) == -dom

        # reversals: runs of frames moving against the dominant direction (1-frame gaps allowed)
        i = a
        while i <= b:
            if against(i) and abs(dd[i]) >= max(min_move, 0.3 * med):
                j, last = i, i
                while j <= b and j - last <= 2:
                    if against(j):
                        last = j
                    j += 1
                if any(o0 <= i <= o1 for o0, o1, _ in osc_spans):
                    i = last + 1
                    continue
                seg_ = dd[i:last + 1]
                back = float(np.sum(seg_[np.sign(seg_) == -dom]))
                nfr = last - i + 1
                ms = float((t[last] - t[max(i - 1, 0)]) * 1000)
                pos = (i - a) / max(1, length - 1)
                pos_end = (last - a) / max(1, length - 1)
                note = ""
                if pos_end >= 0.8 or pos >= 0.8:
                    note = "near gesture end: could be rubber-band bounce at content edge or a snap animation"
                elif pos <= 0.1:
                    note = "at gesture start: could be finger jitter"
                if nfr >= 4:
                    detail = (f"content drifted back {abs(back):.0f}px over {nfr} frames ({ms:.0f} ms, ~{abs(back) / nfr:.0f}px/frame) "
                              f"after moving {'down' if dom > 0 else 'up'}: snap/bounce animation or an offset correction fighting the scroll")
                    sev = 2
                else:
                    detail = (f"content jumped back {abs(back):.0f}px in {nfr} frame(s) (gesture median {med:.0f}px/frame): "
                              f"offset reset or layout shift above the viewport")
                    sev = 3 if abs(back) >= 1.5 * med else 2
                events.append(dict(frame=i, t=float(t[i]), type="reversal", gesture=gi, detail=detail,
                                   note=note, severity=sev if not note else min(sev, 2), frames=nfr, ms=ms))
                i = last + 1
            else:
                i += 1

        # stalls: non-moving frames inside a gesture with motion on both sides
        i = a + 1
        while i < b:
            if not moving[i]:
                j = i
                while j <= b and not moving[j]:
                    j += 1
                fast = max(3 * min_move, 0.5 * med)
                if j <= b and moving[i - 1] and abs(dd[i - 1]) >= fast and abs(dd[j]) >= fast:
                    frames_ = j - i
                    ms = (t[j] - t[i - 1]) * 1000
                    kind = "duplicate frames" if dup[i:j].all() else "no measurable motion"
                    events.append(dict(frame=i, t=float(t[i]), type="stall", gesture=gi,
                                       detail=f"{frames_} frame(s), {ms:.0f} ms without motion ({kind}); "
                                              f"expected ~{med_dt * 1000:.0f} ms/frame",
                                       note="", severity=3 if ms >= 2.5 * med_dt * 1000 else 2,
                                       frames=frames_, ms=float(ms)))
                i = max(j, i + 1)
            else:
                i += 1

        # timing gaps: recorder emitted no frame for a long time while scrolling (Android VFR, dropped frames)
        for i in range(a + 1, b + 1):
            if dt[i] > 1.6 * med_dt and moving[i - 1]:
                expected = med * dt[i] / med_dt
                actual = abs(dd[i])
                if actual >= 0.6 * expected:
                    verdict = ("no frames were produced for the gap, then content caught up to where the scroll "
                               "physics expected it (dropped frames while moving = jank; a recorder-side skip is possible but rare)")
                    sev = 2
                else:
                    verdict = (f"no frames AND no progress: content advanced only {actual:.0f}px instead of ~{expected:.0f}px "
                               "(UI thread blocked, scroll clock stalled = jank)")
                    sev = 3 if dt[i] > 3 * med_dt else 2
                events.append(dict(frame=i, t=float(t[i - 1]), type="timing-gap", gesture=gi,
                                   detail=f"{dt[i] * 1000:.0f} ms between frames (median {med_dt * 1000:.0f} ms); {verdict}",
                                   note="", severity=sev, ms=float(dt[i] * 1000)))

        # jumps: a single frame moving far more than its neighbours (catch-up after a hitch)
        for i in range(a + 3, b + 1):
            if conf[i] < CONF_MIN or not moving[i]:
                continue
            lo, hi = max(a, i - 6), min(b, i + 6)
            neigh = np.abs(dd[lo:hi + 1][moving[lo:hi + 1]])
            neigh = neigh[neigh != abs(dd[i])] if len(neigh) > 1 else neigh
            if len(neigh) < 3:
                continue
            lm = float(np.median(neigh))
            if abs(dd[i]) > max(2.5 * lm, lm + 3 * min_move) and np.sign(dd[i]) == dom:
                before = np.abs(dd[max(a, i - 2):i])
                catch_up = len(before) > 0 and float(before.min()) < 0.5 * lm
                events.append(dict(frame=i, t=float(t[i]), type="jump", gesture=gi,
                                   detail=f"{abs(dd[i]):.0f}px in one frame vs ~{lm:.0f}px around it"
                                          + (" right after a slow/frozen frame (catch-up after a dropped frame)"
                                             if catch_up else " (isolated spike: programmatic scroll or finger acceleration)"),
                                   note="", severity=2 if catch_up else 1))

    # scene changes (navigation, modal, etc.) - context markers, not jank
    for i in range(1, n):
        if conf[i] < 0.02 and mad[i] > 25:
            events.append(dict(frame=i, t=float(t[i]), type="scene-change", gesture=0,
                               detail=f"large content change (mean diff {mad[i]:.0f}/255)", note="", severity=1))

    # merge stall + jump within 3 frames into one 'hitch'
    prio = {"stall": 0, "timing-gap": 0, "jump": 1, "oscillation": 2, "reversal": 2, "scene-change": 3}
    events.sort(key=lambda e: (e["frame"], prio.get(e["type"], 9)))
    merged = []
    for e in events:
        if merged and e["type"] == "jump" and merged[-1]["type"] in ("stall", "timing-gap") \
                and e["frame"] - merged[-1]["frame"] <= 3 + merged[-1].get("frames", 1):
            m = merged[-1]
            m["type"] = "hitch"
            m["detail"] += "; then " + e["detail"].split(" (isolated spike")[0].split(" right after")[0] + " (catch-up)"
            m["severity"] = max(m["severity"], e["severity"])
            continue
        if merged and merged[-1]["frame"] == e["frame"] and merged[-1]["type"] == e["type"]:
            continue
        merged.append(e)
    for k, e in enumerate(merged, 1):
        e["id"] = k
        if e["gesture"]:
            gestures[e["gesture"] - 1]["events"].append(k)

    dup_ratio = float(np.mean(dup[1:])) if n > 1 else 0.0
    changed_no_shift = float(np.mean((mad[1:] > 3.0) & (np.abs(dd[1:]) < min_move))) if n > 1 else 0.0
    gaps = int(np.sum(dt[1:] > 1.6 * med_dt))
    stats = dict(frames=n, effective_fps=round(fps_eff, 2), median_frame_ms=round(med_dt * 1000, 2),
                 duplicate_frame_ratio=round(dup_ratio, 3), timing_gaps=gaps,
                 gestures=len(gestures), min_move_px=min_move,
                 changed_but_no_shift_ratio=round(changed_no_shift, 3),
                 oscillation_spans=len(osc_spans))
    return merged, gestures, stats


def verdict_text(events, gestures, stats):
    jank = [e for e in events if e["type"] in ("stall", "hitch", "timing-gap", "jump") and e["severity"] >= 2]
    major = [e for e in jank if e["severity"] >= 3]
    rev = [e for e in events if e["type"] in ("reversal", "oscillation")]
    strong_rev = [e for e in rev if not e["note"] or e["type"] == "oscillation"]
    osc = [e for e in events if e["type"] == "oscillation"]
    if osc:
        spans = ", ".join(f"{fmt_t(e['t'])}-{fmt_t(e['t'] + e['ms'] / 1000)}" for e in osc[:6])
        kind = "LAYOUT FLICKER" if any(e.get("two_state") for e in osc) else "OSCILLATION"
        amp = max(e.get("amp", 0) for e in osc)
        head = (f"{kind} DETECTED at {spans}: content jumps back and forth by ~{amp:.0f}px "
                f"({'an element of that height is inserted/removed repeatedly' if kind == 'LAYOUT FLICKER' else 'feedback loop between scroll and layout'}). "
                f"This is what a user sees as \"дёргается\"")
        rest, lvl = verdict_text([e for e in events if e["type"] != "oscillation"], gestures, stats) if gestures else ("", "jank")
        return head + (("; scroll gestures: " + rest) if rest else "."), "jank"
    if not gestures:
        hint = ""
        if stats.get("changed_but_no_shift_ratio", 0) > 0.3:
            hint = (" Content changes in %d%% of frames without a dominant shift: a static overlay (sticky header, "
                    "bottom sheet, FAB) may dominate the ROI - retry with bigger --crop-top/--crop-bottom, "
                    "or the motion is not a scroll (animation, fade)." % round(stats["changed_but_no_shift_ratio"] * 100))
        return "NO SCROLL GESTURES DETECTED - nothing moved enough to analyse (try --crop-top/--crop-bottom, --min-move, or check the ROI)." + hint, "none"
    parts = []
    level = "smooth"
    if jank or strong_rev:
        level = "jank"
        parts.append(f"JANK DETECTED in {stats['gestures']} scroll gesture(s): "
                     f"{len(major)} major + {len(jank) - len(major)} minor hitch/stall event(s), "
                     f"{len(strong_rev)} mid-gesture reversal(s)/oscillation(s)")
        worst = max(jank, key=lambda e: e.get("ms", 0), default=None)
        if worst and worst.get("ms"):
            parts.append(f"longest stall ~{worst['ms']:.0f} ms at {fmt_t(worst['t'])}")
        if strong_rev:
            parts.append("reversals at " + ", ".join(fmt_t(e["t"]) for e in strong_rev[:5]))
    else:
        parts.append(f"SCROLL LOOKS SMOOTH across {stats['gestures']} gesture(s) at ~{stats['effective_fps']:.0f} fps")
    if rev and not strong_rev:
        if level == "smooth":
            level = "suspect"
        parts.append(f"{len(rev)} reversal(s) at gesture edges (drift-back at " +
                     ", ".join(fmt_t(e["t"]) for e in rev[:5]) +
                     "): rubber-band bounce if the list was at an edge, otherwise a snap/offset animation fighting the scroll - check the burst sheets")
    if stats["duplicate_frame_ratio"] > 0.3:
        parts.append(f"{stats['duplicate_frame_ratio'] * 100:.0f}% duplicate frames overall (static UI or recorder repeating frames)")
    return "; ".join(parts) + ".", level


# --------------------------------------------------------------------------- drawing
def draw_plot(mo, events, gestures, path, title):
    t, d, dt = mo["t"], mo["d"], mo["dt"]
    Wd, Hd = 1600, 620
    img = Image.new("RGB", (Wd, Hd), "white")
    dr = ImageDraw.Draw(img)
    f = font(13)
    ml, mr, mt = 60, 20, 30
    p1 = (mt, 380)
    p2 = (420, 590)
    tmin, tmax = float(t[0]), float(t[-1])
    if tmax <= tmin:
        tmax = tmin + 1

    def x_of(tt):
        return ml + (tt - tmin) / (tmax - tmin) * (Wd - ml - mr)

    dd = np.nan_to_num(d, nan=0.0)
    amax = max(5.0, float(np.percentile(np.abs(dd), 99)) * 1.2)

    def y1_of(v):
        return (p1[0] + p1[1]) / 2 - v / amax * (p1[1] - p1[0]) / 2

    dtms = np.nan_to_num(dt, nan=0.0) * 1000
    dmax = max(20.0, float(np.percentile(dtms[1:], 99)) * 1.5) if len(dtms) > 1 else 40

    def y2_of(v):
        return p2[1] - v / dmax * (p2[1] - p2[0])

    dr.text((ml, 6), title, fill="black", font=f)
    # gesture bands
    for g in gestures:
        dr.rectangle([x_of(g["start"]), p1[0], x_of(g["end"]), p1[1]], fill=(232, 240, 255))
        dr.text((x_of(g["start"]) + 2, p1[0] + 2), f"G{g['id']}", fill=(60, 90, 180), font=f)
    # axes
    for p in (p1, p2):
        dr.rectangle([ml, p[0], Wd - mr, p[1]], outline=(120, 120, 120))
    dr.line([ml, y1_of(0), Wd - mr, y1_of(0)], fill=(180, 180, 180))
    for v in (amax * 0.5, -amax * 0.5):
        dr.line([ml, y1_of(v), Wd - mr, y1_of(v)], fill=(230, 230, 230))
        dr.text((4, y1_of(v) - 7), f"{v:+.0f}", fill="gray", font=f)
    dr.text((4, y1_of(0) - 7), "0", fill="gray", font=f)
    dr.text((4, p1[0]), "px/frame", fill="gray", font=f)
    dr.text((4, p2[0]), "ms/frame", fill="gray", font=f)
    for v in (dmax * 0.5,):
        dr.line([ml, y2_of(v), Wd - mr, y2_of(v)], fill=(230, 230, 230))
        dr.text((4, y2_of(v) - 7), f"{v:.0f}", fill="gray", font=f)
    med_ms = float(np.nanmedian(dtms[1:])) if len(dtms) > 1 else 16.7
    dr.line([ml, y2_of(med_ms), Wd - mr, y2_of(med_ms)], fill=(170, 220, 170))
    dr.text((4, y2_of(med_ms) - 7), f"{med_ms:.1f}", fill=(60, 140, 60), font=f)
    # ticks
    span = tmax - tmin
    step = 0.5 if span <= 8 else 1 if span <= 30 else 5 if span <= 150 else 15
    tt = math.ceil(tmin / step) * step
    while tt <= tmax:
        x = x_of(tt)
        dr.line([x, p1[1], x, p1[1] + 5], fill="black")
        dr.text((x - 12, p1[1] + 6), fmt_t(tt)[:-2], fill="black", font=f)
        dr.line([x, p2[1], x, p2[1] + 5], fill="black")
        dr.text((x - 12, p2[1] + 6), fmt_t(tt)[:-2], fill="black", font=f)
        tt += step
    # series
    pts1 = [(x_of(t[i]), y1_of(max(-amax, min(amax, dd[i])))) for i in range(len(t))]
    if len(pts1) > 1:
        dr.line(pts1, fill=(30, 100, 220), width=2)
    for i in np.where(np.isnan(d))[0]:
        dr.point((x_of(t[i]), y1_of(0)), fill="orange")
    pts2 = [(x_of(t[i]), y2_of(min(dmax, dtms[i]))) for i in range(1, len(t))]
    if len(pts2) > 1:
        dr.line(pts2, fill=(90, 90, 90), width=1)
    colors = dict(stall=(220, 30, 30), hitch=(220, 30, 30), reversal=(200, 0, 200), oscillation=(120, 0, 160),
                  jump=(230, 130, 0), **{"timing-gap": (0, 150, 150), "scene-change": (150, 150, 150)})
    for e in events:
        x = x_of(e["t"])
        c = colors.get(e["type"], "black")
        dr.line([x, p1[0], x, p2[1]], fill=c, width=1)
        dr.text((x + 2, p1[0] + 14 + (e["id"] % 4) * 13), f"E{e['id']}", fill=c, font=f)
    legend = "E# = event (red stall/hitch, magenta reversal, orange jump, teal timing gap, gray scene change); " \
             "top: scroll delta per frame (+ = scrolling down); bottom: frame interval"
    dr.text((ml, Hd - 18), legend, fill="black", font=f)
    img.save(path, quality=90)


def contact_sheet(tiles, labels, cols, path, highlight=None, tile_w=320, label_h=22, mark=None):
    """tiles: list of PIL images; labels: list of str; highlight: set of indices to outline red."""
    if not tiles:
        return
    th = max(int(round(im.height * tile_w / im.width)) for im in tiles)
    rows = math.ceil(len(tiles) / cols)
    sheet = Image.new("RGB", (cols * tile_w, rows * (th + label_h)), (24, 24, 24))
    dr = ImageDraw.Draw(sheet)
    f = font(14)
    for k, (im, lab) in enumerate(zip(tiles, labels)):
        r, c = divmod(k, cols)
        x, y = c * tile_w, r * (th + label_h)
        im2 = im.convert("RGB").resize((tile_w, int(round(im.height * tile_w / im.width))), Image.LANCZOS)
        dr.rectangle([x, y, x + tile_w, y + label_h], fill=(40, 40, 40))
        dr.text((x + 6, y + 4), lab, fill=(255, 255, 255), font=f)
        sheet.paste(im2, (x, y + label_h))
        if highlight and k in highlight:
            dr.rectangle([x + 1, y + label_h + 1, x + tile_w - 2, y + label_h + im2.height - 2],
                         outline=(255, 40, 40), width=4)
    sheet.save(path, quality=88)
    return sheet.size


def extract_images(video, out_dir, vf, start=None, duration=None, max_frames=None, prefix="f"):
    """Extract frames as JPEG with exact timestamps via showinfo. Returns [(path, pts_time)]."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(f"{prefix}_*.jpg"):
        old.unlink()
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-loglevel", "info", "-y"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", video]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-an", "-sn", "-dn", "-vf", vf + ",showinfo"] + passthrough_flags()
    if max_frames:
        cmd += ["-frames:v", str(max_frames)]
    cmd += ["-q:v", "3", str(out_dir / f"{prefix}_%04d.jpg")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    pts = parse_showinfo_pts(r.stderr)
    files = sorted(out_dir.glob(f"{prefix}_*.jpg"))
    t0 = start or 0.0
    res = []
    for k, p in enumerate(files):
        tt = pts[k] + t0 if k < len(pts) else None
        res.append((p, tt))
    return res


# --------------------------------------------------------------------------- audio
def do_audio(video, meta, out):
    if not meta["has_audio"]:
        return "no audio stream"
    wav = out / "audio.wav"
    r = run(["ffmpeg", "-y", "-v", "error", "-i", video, "-vn", "-ac", "1", "-ar", "16000", str(wav)])
    if r.returncode:
        return f"audio extraction failed: {r.stderr.strip()[:200]}"
    vd = run(["ffmpeg", "-v", "info", "-i", str(wav), "-af", "volumedetect", "-f", "null", "-"])
    m = re.search(r"mean_volume:\s*(-?[0-9.]+) dB", vd.stderr)
    mean_db = float(m.group(1)) if m else None
    if mean_db is not None and mean_db < -50:
        return f"audio is essentially silent (mean {mean_db:.0f} dB) - no narration"
    for tool in ("whisper", "mlx_whisper", "whisper-cli", "whisper-cpp"):
        exe = shutil.which(tool)
        if not exe:
            continue
        if tool in ("whisper", "mlx_whisper"):
            cmd = [exe, str(wav), "--output_format", "txt", "--output_dir", str(out)]
            if tool == "whisper":
                cmd += ["--model", "base"]
        else:
            cmd = [exe, "-f", str(wav), "-otxt", "-of", str(out / "audio")]
        r = subprocess.run(cmd, capture_output=True, text=True)
        txt = out / "audio.txt"
        if txt.exists():
            return f"transcript via {tool}: {txt}"
        return f"{tool} ran but produced no transcript ({r.stderr.strip()[-200:]})"
    return (f"audio present (mean {mean_db:.0f} dB) but no transcriber installed; "
            "optional: `pip3 install openai-whisper` (or `brew install whisper-cpp`) and re-run")


# --------------------------------------------------------------------------- analyze
def cmd_analyze(args):
    need("ffmpeg")
    video = str(Path(args.video).expanduser().resolve())
    if not os.path.exists(video):
        sys.exit(f"file not found: {video}")
    meta = probe(video)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(video).stem)[:60]
    out = Path(args.out).expanduser() if args.out else Path.home() / ".cache" / "claude-video" / stem
    out.mkdir(parents=True, exist_ok=True)
    for stale in list(out.glob("event_*.jpg")) + list(out.glob("sheet_*.jpg")):
        stale.unlink()
    if (out / "frames_events").exists():
        shutil.rmtree(out / "frames_events")
    portrait = meta["height"] >= meta["width"]
    dur = meta["duration"]
    seg_start = args.start or 0.0
    seg_end = args.end or dur
    print(f"[probe] {meta['width']}x{meta['height']} {meta['codec']} {meta['avg_fps']:.2f} fps "
          f"{dur:.2f}s audio={'yes' if meta['has_audio'] else 'no'} -> {out}", file=sys.stderr)

    # ---- motion
    events, gestures, stats, mo, verdict, level = [], [], {}, None, "motion analysis skipped", "skipped"
    if not args.no_motion:
        est_frames = (seg_end - seg_start) * (meta["avg_fps"] or 30)
        if est_frames > 9000:
            print(f"[warn] ~{est_frames:.0f} frames to analyse; consider --start/--end", file=sys.stderr)
        mo, err = analyze_motion(video, meta, out, args)
        if err:
            print(f"[warn] {err}", file=sys.stderr)
        else:
            min_move = args.min_move if args.min_move else max(2.0, 1.5 / mo["scale"])
            events, gestures, stats = detect_events(mo, min_move)
            verdict, level = verdict_text(events, gestures, stats)
            with open(out / "motion.csv", "w") as fh:
                fh.write("frame,t,dt_ms,d_offset_px,dx_px,conf,mean_abs_diff,duplicate\n")
                for i in range(mo["n"]):
                    fh.write(f"{i},{mo['t'][i]:.4f},{(mo['dt'][i] * 1000) if i else 0:.2f},"
                             f"{mo['d'][i]:.1f},{mo['dx'][i]:.1f},{mo['conf'][i]:.3f},{mo['mad'][i]:.2f},{int(mo['dup'][i])}\n")
            draw_plot(mo, events, gestures, out / "motion.png",
                      f"{Path(video).name}  scroll delta per frame / frame interval")

    # ---- overview sheets
    overview = []
    sheets = []
    if not args.no_sheets:
        span = max(0.001, seg_end - seg_start)
        fps = min(args.sheet_fps, args.max_frames / span)
        tile_w = args.tile_width or (300 if portrait else 400)
        cols, rows = (5, 2) if portrait else (4, 3)
        if args.grid:
            cols, rows = [int(x) for x in args.grid.lower().split("x")]
        per = cols * rows
        frames = extract_images(video, out / "frames_overview",
                                f"fps={fps:.4f},scale={tile_w}:-2", seg_start if args.start else None,
                                (seg_end - seg_start) if args.end else None, args.max_frames, prefix="ov")
        for s in range(0, len(frames), per):
            chunk = frames[s:s + per]
            tiles = [Image.open(p) for p, _ in chunk]
            labels = [f"#{s + k + 1}  t={fmt_t(tt)}" for k, (_, tt) in enumerate(chunk)]
            path = out / f"sheet_{s // per + 1:02d}.jpg"
            contact_sheet(tiles, labels, cols, path, tile_w=tile_w)
            sheets.append((path, chunk[0][1], chunk[-1][1]))
        overview = frames

    # ---- event burst sheets
    burst_paths = {}
    if mo is not None and events and not args.no_sheets:
        tile_w = args.tile_width or (300 if portrait else 400)
        ranked = sorted([e for e in events if e["type"] != "scene-change" and e["severity"] >= 2],
                        key=lambda e: (-e["severity"], e["t"]))[:args.max_events]
        for e in sorted(ranked, key=lambda e: e["t"]):
            i = e["frame"]
            lo = max(0, i - 4)
            hi = min(mo["n"] - 1, i + max(5, e.get("frames", 1) + 3), lo + 14)
            t_lo = float(mo["t"][lo]) - 0.0005
            dur_ = float(mo["t"][hi]) - t_lo + 0.5 / max(1.0, stats.get("effective_fps", 30))
            frames = extract_images(video, out / "frames_events", f"scale={tile_w}:-2", t_lo, dur_,
                                    hi - lo + 1, prefix=f"ev{e['id']:02d}")
            tiles, labels, hl = [], [], set()
            for k, (p, tt) in enumerate(frames):
                j = int(np.argmin(np.abs(mo["t"] - tt))) if tt is not None else lo + k
                dv = mo["d"][j]
                dv_s = "nan" if np.isnan(dv) else f"{dv:+.0f}"
                labels.append(f"f{j} t={fmt_t(mo['t'][j])} d={dv_s}px dt={mo['dt'][j] * 1000 if j else 0:.0f}ms")
                tiles.append(Image.open(p))
                span = max(1, e.get("frames", 1)) + (1 if e["type"] == "timing-gap" else 0)
                if i <= j <= i + span - 1:
                    hl.add(k)
            cols = 5 if portrait else 4
            path = out / f"event_{e['id']:02d}_{e['type']}_{fmt_t(e['t']).replace(':', 'm')}s.jpg"
            contact_sheet(tiles, labels, cols, path, highlight=hl, tile_w=tile_w)
            burst_paths[e["id"]] = path
            e["sheet"] = str(path)

    # ---- audio
    audio_note = "skipped" if args.no_audio else do_audio(video, meta, out)

    # ---- report
    rep = []
    rep.append(f"# Video analysis: {Path(video).name}\n")
    rep.append("## Metadata\n")
    rep.append("| field | value |\n|---|---|")
    for k in ("path", "size_mb", "width", "height", "rotation", "codec", "duration", "avg_fps", "r_fps", "nb_frames", "has_audio"):
        rep.append(f"| {k} | {meta[k]} |")
    if stats:
        for k, v in stats.items():
            rep.append(f"| {k} | {v} |")
    if args.start or args.end:
        rep.append(f"| analysed range | {fmt_t(seg_start)} - {fmt_t(seg_end)} |")
    rep.append("")
    rep.append(f"## Verdict\n\n**{verdict}**\n")
    if mo is not None:
        rep.append(f"Motion plot: `{out / 'motion.png'}` (view it). Per-frame data: `{out / 'motion.csv'}`. "
                   f"ROI rows {mo['roi'][0]}-{mo['roi'][1]} of {mo['H']} (crop-top {args.crop_top}, crop-bottom {args.crop_bottom}), "
                   f"analysis width {mo['W']}px (1 unit = {1 / mo['scale']:.1f} original px).\n")
    if gestures:
        rep.append("## Scroll gestures\n")
        rep.append("| # | start | end | frames | total px | direction | median px/frame | px/s | events |\n|---|---|---|---|---|---|---|---|---|")
        for g in gestures:
            ev = ", ".join(f"E{x}" for x in g["events"]) or "-"
            rep.append(f"| G{g['id']} | {fmt_t(g['start'])} | {fmt_t(g['end'])} | {g['frames']} | {g['total_px']:+.0f} | "
                       f"{g['direction']} | {g['median_px_per_frame']:.1f} | {g['px_per_s']:.0f} | {ev} |")
        rep.append("")
    if events:
        rep.append("## Events (chronological)\n")
        rep.append("| # | t | frame | type | sev | detail | burst sheet |\n|---|---|---|---|---|---|---|")
        for e in events:
            note = f" _{e['note']}_" if e.get("note") else ""
            sheet = f"`{e['sheet']}`" if e.get("sheet") else "-"
            rep.append(f"| E{e['id']} | {fmt_t(e['t'])} | {e['frame']} | {e['type']} | {e['severity']} | {e['detail']}{note} | {sheet} |")
        rep.append("")
        rep.append("Event types: **stall** = frames without motion mid-gesture (dropped/duplicated frames = the app did not "
                   "produce new frames); **hitch** = stall followed by a catch-up jump (classic dropped frame); "
                   "**timing-gap** = the recording itself has no frame for a while (VFR recorder: screen not updated); "
                   "**jump** = one frame moved much farther than neighbours; **reversal** = content moved against the "
                   "scroll direction mid-gesture (jump back = offset reset / layout shift; drift back over several frames = "
                   "snap/bounce animation or an offset correction fighting the scroll); **oscillation** = content shakes back and "
                   "forth (feedback loop between scroll and layout/offset updates); "
                   "**scene-change** = navigation or big UI change (context only).\n")
    if sheets:
        rep.append("## Overview contact sheets (view in order)\n")
        for p, a, b in sheets:
            rep.append(f"- `{p}`  ({fmt_t(a)} - {fmt_t(b)})")
        rep.append("")
    rep.append(f"## Audio\n\n{audio_note}\n")
    rep.append("## Dig deeper\n")
    rep.append("```bash")
    rep.append(f"# every frame of a time window as separate images (add --sheet for a grid):")
    rep.append(f"python3 {Path(__file__).resolve()} window \"{video}\" START END --fps 0 --width 540 --sheet")
    rep.append(f"# re-run motion analysis on a sub-range or with a different ROI:")
    rep.append(f"python3 {Path(__file__).resolve()} analyze \"{video}\" --start S --end E --crop-top 0.15 --crop-bottom 0.15")
    rep.append("```\n")
    (out / "report.md").write_text("\n".join(rep), encoding="utf-8")

    summary = dict(video=video, out=str(out), meta=meta, stats=stats, verdict=verdict, level=level,
                   gestures=gestures, events=events, sheets=[str(p) for p, _, _ in sheets],
                   motion_plot=str(out / "motion.png") if mo is not None else None, audio=audio_note)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float, ensure_ascii=False), encoding="utf-8")

    # ---- stdout for the agent
    print(f"VIDEO   {video}")
    print(f"META    {meta['width']}x{meta['height']} {meta['codec']} {meta['avg_fps']:.2f}fps {dur:.2f}s "
          f"audio={'yes' if meta['has_audio'] else 'no'}")
    if stats:
        print(f"MOTION  frames={stats['frames']} eff_fps={stats['effective_fps']} dup_ratio={stats['duplicate_frame_ratio']} "
              f"gestures={stats['gestures']} timing_gaps={stats['timing_gaps']}")
    print(f"VERDICT {verdict}")
    for g in gestures:
        print(f"  G{g['id']}: {fmt_t(g['start'])}-{fmt_t(g['end'])} {g['total_px']:+.0f}px {g['direction']} "
              f"{g['median_px_per_frame']:.1f}px/frame events={g['events'] or '-'}")
    for e in events:
        if e["type"] == "scene-change":
            continue
        print(f"  E{e['id']} {fmt_t(e['t'])} {e['type']:<10} {e['detail']}" + (f"  [{e['note']}]" if e.get("note") else ""))
    print(f"AUDIO   {audio_note}")
    print(f"REPORT  {out / 'report.md'}")
    if mo is not None:
        print(f"PLOT    {out / 'motion.png'}")
    for p, a, b in sheets:
        print(f"SHEET   {p}  {fmt_t(a)}-{fmt_t(b)}")
    for e in events:
        if e.get("sheet"):
            print(f"BURST   E{e['id']} {e['sheet']}")
    print("NEXT    read report.md, view PLOT, view SHEET files in order, then BURST sheets for each event.")


# --------------------------------------------------------------------------- window
def cmd_window(args):
    need("ffmpeg")
    video = str(Path(args.video).expanduser().resolve())
    meta = probe(video)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(video).stem)[:60]
    out = Path(args.out).expanduser() if args.out else Path.home() / ".cache" / "claude-video" / stem / f"window_{args.start:.2f}-{args.end:.2f}"
    vf = f"scale={args.width}:-2"
    if args.fps and args.fps > 0:
        vf = f"fps={args.fps}," + vf
    frames = extract_images(video, out, vf, args.start, args.end - args.start, args.max_frames, prefix="w")
    print(f"WINDOW  {fmt_t(args.start)}-{fmt_t(args.end)}  {len(frames)} frames -> {out}")
    for k, (p, tt) in enumerate(frames):
        print(f"  {p}  t={fmt_t(tt)}")
    if args.sheet and frames:
        portrait = meta["height"] >= meta["width"]
        cols = 5 if portrait else 4
        per = cols * (2 if portrait else 3)
        tw = 300 if portrait else 400
        for s in range(0, len(frames), per):
            chunk = frames[s:s + per]
            path = out / f"window_sheet_{s // per + 1:02d}.jpg"
            contact_sheet([Image.open(p) for p, _ in chunk],
                          [f"#{s + k + 1} t={fmt_t(tt)}" for k, (_, tt) in enumerate(chunk)], cols, path, tile_w=tw)
            print(f"SHEET   {path}")


def cmd_rows(args):
    video = str(Path(args.video).expanduser().resolve())
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(video).stem)[:60]
    out = Path(args.out).expanduser() if args.out else Path.home() / ".cache" / "claude-video" / stem
    csv = out / "motion.csv"
    if not csv.exists():
        sys.exit(f"{csv} not found - run analyze first")
    lines = csv.read_text().splitlines()
    print(lines[0])
    for ln in lines[1:]:
        parts = ln.split(",")
        try:
            tt = float(parts[1])
        except ValueError:
            continue
        if args.start <= tt <= args.end:
            print(ln)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="full pipeline")
    a.add_argument("video")
    a.add_argument("--out", help="output dir (default ~/.cache/claude-video/<name>)")
    a.add_argument("--start", type=float, help="analyse from this second")
    a.add_argument("--end", type=float, help="analyse until this second")
    a.add_argument("--width", type=int, default=480, help="analysis width in px (default 480)")
    a.add_argument("--crop-top", type=float, default=0.10, help="ignore this top fraction (status bar/header)")
    a.add_argument("--crop-bottom", type=float, default=0.10, help="ignore this bottom fraction (tab bar/home indicator)")
    a.add_argument("--min-move", type=float, default=0.0, help="min px/frame (original px) to count as motion")
    a.add_argument("--sheet-fps", type=float, default=2.0, help="overview sampling fps cap (default 2)")
    a.add_argument("--max-frames", type=int, default=60, help="overview frames cap (default 60)")
    a.add_argument("--grid", help="overview grid CxR, e.g. 5x2")
    a.add_argument("--tile-width", type=int, default=0, help="tile width px (default 300 portrait / 400 landscape)")
    a.add_argument("--max-events", type=int, default=8, help="max burst sheets for events")
    a.add_argument("--no-motion", action="store_true")
    a.add_argument("--no-sheets", action="store_true")
    a.add_argument("--no-audio", action="store_true")
    a.set_defaults(fn=cmd_analyze)

    w = sub.add_parser("window", help="dump frames of a time window")
    w.add_argument("video")
    w.add_argument("start", type=float)
    w.add_argument("end", type=float)
    w.add_argument("--fps", type=float, default=0, help="sampling fps; 0 = every frame (default)")
    w.add_argument("--width", type=int, default=540)
    w.add_argument("--max-frames", type=int, default=120)
    w.add_argument("--out")
    w.add_argument("--sheet", action="store_true", help="also build contact sheets")
    w.set_defaults(fn=cmd_window)

    r = sub.add_parser("rows", help="print motion.csv rows for a time range (after analyze)")
    r.add_argument("video")
    r.add_argument("start", type=float)
    r.add_argument("end", type=float)
    r.add_argument("--out")
    r.set_defaults(fn=cmd_rows)

    p = sub.add_parser("probe", help="metadata as JSON")
    p.add_argument("video")
    p.set_defaults(fn=lambda a_: print(json.dumps(probe(str(Path(a_.video).expanduser().resolve())), indent=1, ensure_ascii=False)))

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
