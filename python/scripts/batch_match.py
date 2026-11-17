#!/usr/bin/env python3
"""Play many GTP games at once between two engines running in batch GTP mode.

Batch GTP mode (p3achygo `gtp --batch`, KataGo `gtp -batch`, Leela Zero
`--gtp-batch N`): the GTP id of a command is the game id, commands for
different games run concurrently inside one engine process sharing one
batched network, and responses `=id ...` come back in completion order.
This script opens one process per side and drives N games through them.

Example:
  python batch_match.py --games 32 --concurrency 16 \\
    --a-name p3achygo --a "$P3/build/gtp --model_path m.trt --config c.cfg --n 400 --batch --batch_games 16" \\
    --b-name katago   --b "katago gtp -batch -model k.bin.gz -config gtp.cfg -override-config maxVisits=400,batchMaxGames=16" \\
    --sgf-dir sgfs --out results.json

Colors alternate per game. Visit limits (not clocks) belong in the engine
command lines: with shared batching, per-move wall time depends on the other
games.
"""
import argparse
import json
import math
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime


class GtpError(Exception):
    pass


class BatchEngine:
    """One engine process in batch GTP mode; routes responses by game id."""

    def __init__(self, name, cmd, stderr_path):
        self.name = name
        self.stderr = open(stderr_path, "w")
        self.proc = subprocess.Popen(
            shlex.split(cmd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
            bufsize=1,
        )
        self.write_lock = threading.Lock()
        self.queues = {}  # game id (None for global) -> Queue of (ok, text)
        self.queues_lock = threading.Lock()
        self.dead = False
        self.reader = threading.Thread(target=self._read_loop, daemon=True)
        self.reader.start()

    def _queue(self, gid):
        with self.queues_lock:
            return self.queues.setdefault(gid, queue.Queue())

    def _read_loop(self):
        block = []
        for line in self.proc.stdout:
            line = line.rstrip("\n")
            if line == "":
                if block:
                    self._dispatch(block)
                    block = []
                continue
            block.append(line)
        if block:
            self._dispatch(block)
        self.dead = True
        # Wake every waiter.
        with self.queues_lock:
            for q in self.queues.values():
                q.put((False, "engine exited"))

    def _dispatch(self, block):
        m = re.match(r"^([=?])(\d*)\s?(.*)$", block[0])
        if not m:
            sys.stderr.write(f"[{self.name}] unparseable response: {block[0]!r}\n")
            return
        ok = m.group(1) == "="
        gid = int(m.group(2)) if m.group(2) else None
        text = "\n".join([m.group(3)] + block[1:]).strip()
        self._queue(gid).put((ok, text))

    def send(self, gid, cmd, timeout=None):
        """Sends `cmd` for game `gid` (None = global) and waits for its reply."""
        q = self._queue(gid)
        line = f"{gid} {cmd}\n" if gid is not None else f"{cmd}\n"
        with self.write_lock:
            if self.dead:
                raise GtpError(f"{self.name} exited")
            self.proc.stdin.write(line)
            self.proc.stdin.flush()
        try:
            ok, text = q.get(timeout=timeout)
        except queue.Empty:
            raise GtpError(f"{self.name}: timeout on '{line.strip()}'")
        if not ok:
            raise GtpError(f"{self.name}: '{line.strip()}' -> ? {text}")
        return text

    def close(self):
        try:
            with self.write_lock:
                self.proc.stdin.write("quit\n")
                self.proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.stderr.close()


SCORE_RE = re.compile(r"^([BW])\+(\d+(?:\.\d+)?)$", re.I)
P3_SCORE_RE = re.compile(r"^B:\s*(-?\d+(?:\.\d+)?),\s*W:\s*(-?\d+(?:\.\d+)?)$")


def parse_score(text):
    """Returns (winner 'B'/'W'/'0', margin) from a final_score reply."""
    text = text.strip()
    if text == "0":
        return "0", 0.0
    m = SCORE_RE.match(text)
    if m:
        return m.group(1).upper(), float(m.group(2))
    m = P3_SCORE_RE.match(text)  # p3achygo: "B: 2, W: 8.5"
    if m:
        b, w = float(m.group(1)), float(m.group(2))
        if b == w:
            return "0", 0.0
        return ("B", b - w) if b > w else ("W", w - b)
    raise GtpError(f"cannot parse score {text!r}")


def gtp_to_sgf(move):
    move = move.lower()
    if move == "pass":
        return ""
    col = "abcdefghjklmnopqrst".index(move[0])
    row = 19 - int(move[1:])
    return "abcdefghijklmnopqrs"[col] + "abcdefghijklmnopqrs"[row]


def write_sgf(path, black, white, komi, moves, result, date):
    sgf = (
        f"(;GM[1]FF[4]SZ[19]KM[{komi}]PB[{black}]PW[{white}]"
        f"RE[{result}]DT[{date}]RU[Chinese]"
    )
    for color, move in moves:
        sgf += f";{color}[{gtp_to_sgf(move)}]"
    sgf += ")\n"
    with open(path, "w") as f:
        f.write(sgf)


def play_game(gid, engines, black_idx, komi, max_moves, scorer_idx, args):
    """Plays one game; engines[black_idx] is black. Returns a result dict."""
    black = engines[black_idx]
    white = engines[1 - black_idx]
    by_color = {"B": black, "W": white}
    t0 = time.time()
    for e in engines:
        e.send(gid, "boardsize 19", timeout=args.timeout)
        e.send(gid, "clear_board", timeout=args.timeout)
        e.send(gid, f"komi {komi}", timeout=args.timeout)
    moves, passes, winner, how = [], 0, None, ""
    color = "B"
    try:
        while True:
            mover, other = by_color[color], by_color["W" if color == "B" else "B"]
            move = mover.send(
                gid, f"genmove {color.lower()}", timeout=args.timeout
            ).strip()
            if move.lower() == "resign":
                winner, how = ("W" if color == "B" else "B"), "Resign"
                break
            try:
                other.send(gid, f"play {color} {move}", timeout=args.timeout)
            except GtpError as e:
                # The opponent rejected the move: count it against the mover.
                winner, how = ("W" if color == "B" else "B"), f"Illegal({move}; {e})"
                break
            moves.append((color, move))
            passes = passes + 1 if move.lower() == "pass" else 0
            if passes >= 2:
                scorer = engines[scorer_idx]
                scores = {}
                for e in engines:
                    try:
                        scores[e.name] = e.send(
                            gid, "final_score", timeout=args.timeout
                        )
                    except GtpError as err:
                        scores[e.name] = f"error: {err}"
                winner, margin = parse_score(scores[scorer.name])
                how = f"{margin}" if winner != "0" else "Draw"
                break
            if len(moves) >= max_moves:
                winner, how = "0", "MaxMoves"
                break
            color = "W" if color == "B" else "B"
    finally:
        for e in engines:
            try:
                e.send(gid, "quit", timeout=args.timeout)
            except GtpError:
                pass
    result = "Draw" if winner == "0" else f"{winner}+{how}"
    date = datetime.now().strftime("%Y-%m-%d")
    if args.sgf_dir:
        write_sgf(
            os.path.join(args.sgf_dir, f"game_{gid:04d}.sgf"),
            black.name,
            white.name,
            komi,
            moves,
            result,
            date,
        )
    return {
        "game": gid,
        "black": black.name,
        "white": white.name,
        "winner": winner,
        "winner_name": None if winner == "0" else by_color[winner].name,
        "result": result,
        "moves": len(moves),
        "seconds": round(time.time() - t0, 1),
    }


def summarize(results, names):
    wins = {n: 0 for n in names}
    draws = 0
    for r in results:
        if r.get("winner_name"):
            wins[r["winner_name"]] += 1
        elif r.get("winner") == "0":
            draws += 1
    n = len(results)
    a = names[0]
    p = wins[a] / n if n else 0.0
    se = math.sqrt(p * (1 - p) / n) if n else 0.0
    elo = (
        400 * math.log10(p / (1 - p))
        if 0 < p < 1
        else float("inf") if p == 1 else float("-inf")
    )
    return {
        "games": n,
        "wins": wins,
        "draws": draws,
        f"{a}_winrate": round(p, 4),
        "winrate_95ci": round(1.96 * se, 4),
        f"{a}_elo_vs_{names[1]}": round(elo, 1) if math.isfinite(elo) else str(elo),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--a", required=True, help="engine A command line (batch GTP mode)")
    ap.add_argument("--b", required=True, help="engine B command line (batch GTP mode)")
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-name", default="B")
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="games in flight at once; must not exceed either engine's max games",
    )
    ap.add_argument("--komi", type=float, default=7.5)
    ap.add_argument("--max-moves", type=int, default=600)
    ap.add_argument(
        "--scorer",
        choices=["a", "b"],
        default="b",
        help="whose final_score decides a two-pass game",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="seconds to wait for any single GTP reply",
    )
    ap.add_argument("--sgf-dir", default=None)
    ap.add_argument("--out", default=None, help="results JSON path")
    ap.add_argument("--log-dir", default=".", help="where engine stderr logs go")
    args = ap.parse_args()

    if args.sgf_dir:
        os.makedirs(args.sgf_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    engines = [
        BatchEngine(
            args.a_name, args.a, os.path.join(args.log_dir, f"{args.a_name}.stderr.log")
        ),
        BatchEngine(
            args.b_name, args.b, os.path.join(args.log_dir, f"{args.b_name}.stderr.log")
        ),
    ]
    names = [e.name for e in engines]
    for e in engines:
        print(
            f"{e.name}: {e.send(None, 'name', timeout=args.timeout)} {e.send(None, 'version', timeout=args.timeout)}",
            flush=True,
        )
    scorer_idx = 0 if args.scorer == "a" else 1

    results, failures = [], []
    lock = threading.Lock()
    sem = threading.Semaphore(args.concurrency)
    t0 = time.time()

    def run(gid):
        with sem:
            try:
                r = play_game(
                    gid,
                    engines,
                    black_idx=(gid - 1) % 2,
                    komi=args.komi,
                    max_moves=args.max_moves,
                    scorer_idx=scorer_idx,
                    args=args,
                )
                with lock:
                    results.append(r)
                    done = len(results)
                print(
                    f"[{done}/{args.games}] game {gid}: {r['black']}(B) vs {r['white']}(W) -> "
                    f"{r['result']} in {r['moves']} moves, {r['seconds']}s",
                    flush=True,
                )
            except Exception as e:  # keep the run going
                with lock:
                    failures.append({"game": gid, "error": str(e)})
                print(f"game {gid} FAILED: {e}", flush=True)

    threads = [
        threading.Thread(target=run, args=(gid,)) for gid in range(1, args.games + 1)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for e in engines:
        e.close()

    summary = summarize(results, names)
    summary["failures"] = failures
    summary["wall_seconds"] = round(time.time() - t0, 1)
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {
                    "summary": summary,
                    "games": sorted(results, key=lambda r: r["game"]),
                    "args": vars(args),
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
