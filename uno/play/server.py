"""`uv run uno play [--port 8765] [--ruleset FULL_UNO] [--ckpt PATH[=LABEL] ...] [--cards DIR]`

Serves a one-page UNO table on localhost where you play against any rulebot or a
loaded actor export. Standard library only: the engine runs here, the page only
animates what the server reports.

Each request returns the *steps* since the human's last decision -- one per engine
action, with the public events it produced and the view after it -- so the page can
replay the bot's moves one by one. Opponent actions are sampled from the agent's
distribution; the demo is not an evaluation and uses no registered seeds.

Card sprites are optional: put files named like `red_7.png`, `blue_skip.png`,
`green_reverse.png`, `yellow_draw2.png`, `wild.png`, `wild_draw4.png` and `back.png`
(png/svg/webp/jpg) in `uno/play/static/cards/` or pass `--cards DIR`. Missing ones
are drawn in CSS.
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import uuid
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from uno.game.engine import Phase, State, legal_actions, new_game, step
from uno.game.rules import NAMED, Kind, Play, Ruleset
from uno.play.opponents import Opponent, Player, checkpoint, rulebots

STATIC = Path(__file__).parent / "static"
COLOR_NAMES = ("red", "yellow", "green", "blue", "purple", "orange")
KIND_NAMES = {Kind.NUMBER: "number", Kind.SKIP: "skip", Kind.REVERSE: "reverse",
              Kind.DRAW_TWO: "draw2", Kind.WILD: "wild", Kind.WILD_DRAW_FOUR: "wild4"}
SPRITE_EXT = (".png", ".svg", ".webp", ".jpg", ".jpeg")
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".png": "image/png", ".svg": "image/svg+xml",
                 ".webp": "image/webp", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


def sprite_name(rs: Ruleset, card: int) -> str:
    ct = rs.card_types[card]
    if ct.kind is Kind.WILD:
        return "wild"
    if ct.kind is Kind.WILD_DRAW_FOUR:
        return "wild_draw4"
    color = COLOR_NAMES[ct.color]
    return f"{color}_{ct.rank}" if ct.kind is Kind.NUMBER else f"{color}_{KIND_NAMES[ct.kind]}"


class Game:
    def __init__(self, rs: Ruleset, opponent: Opponent, human: int, seed: int) -> None:
        self.rs, self.opponent, self.human, self.seed = rs, opponent, human, seed
        self.state: State = new_game(rs, seed)
        self.bot: Player = opponent.make(human ^ 1)
        self.rng = random.Random(seed ^ 0x5EED)

    def who(self, seat: int) -> str:
        return "you" if seat == self.human else "bot"

    def view(self) -> dict:
        s, rs, me = self.state, self.rs, self.human
        legal = legal_actions(s) if s.player == me and not s.terminal else ()
        plays = [{"action": a, "card": rs.actions[a].card, "color": rs.actions[a].color}
                 for a in legal if isinstance(rs.actions[a], Play)]
        return {
            "hand": list(s.hands[me]),
            "opp_count": len(s.hands[me ^ 1]),
            "top": s.top,
            "color": s.current_color,
            "pile": len(s.draw_pile),
            "discard": len(s.discard),
            "turn": s.turn,
            "to_move": self.who(s.player),
            "after_draw": s.phase is Phase.AFTER_DRAW and s.player == me,
            "drawn": s.drawn if s.player == me else None,
            "plays": plays,
            "draw_action": rs.draw_action if rs.draw_action in legal else None,
            "pass_action": rs.pass_action if rs.pass_action in legal else None,
            "over": s.terminal,
            "end": s.end,
            "winner": None if s.winner is None else self.who(s.winner),
            "opp_hand": list(s.hands[me ^ 1]) if s.terminal else None,
        }

    def _apply(self, action: int, prob: float | None) -> dict:
        before = self.state
        seat = before.player
        self.state = step(before, action)
        events = []
        own_before = len(before.own_draws[self.human])
        for e in self.state.public[len(before.public):]:
            match e:
                case ("play", p, card, color):
                    events.append({"type": "play", "who": self.who(p), "card": card, "color": color})
                case ("draw", p, n):
                    ev = {"type": "draw", "who": self.who(p), "n": n}
                    if p == self.human:
                        ev["cards"] = list(self.state.own_draws[p][own_before:own_before + n])
                        own_before += n
                    events.append(ev)
                case ("reshuffle",):
                    events.append({"type": "reshuffle"})
        if action == self.rs.pass_action:
            events.append({"type": "pass", "who": self.who(seat)})
        return {"who": self.who(seat), "action": action, "prob": prob, "events": events, "view": self.view()}

    def act(self, action: int) -> list[dict]:
        if self.state.terminal or self.state.player != self.human:
            raise ValueError("not your turn")
        if action not in legal_actions(self.state):
            raise ValueError(f"illegal action {action}")
        return [self._apply(action, None)] + self.run_bot()

    def run_bot(self) -> list[dict]:
        steps = []
        while not self.state.terminal and self.state.player != self.human:
            probs = self.bot.policy(self.state)
            actions = list(probs)
            a = self.rng.choices(actions, weights=[probs[x] for x in actions])[0]
            steps.append(self._apply(a, probs[a]))
        return steps


class App:
    def __init__(self, rs: Ruleset, cards_dir: Path | None) -> None:
        self.rs = rs
        self.lock = threading.Lock()
        self.opponents: dict[str, Opponent] = {o.id: o for o in rulebots(rs)}
        self.games: dict[str, Game] = {}
        self.cards_dirs = [d for d in (cards_dir, STATIC / "cards") if d is not None and d.is_dir()]

    def load(self, path: str, label: str | None = None) -> Opponent:
        opp = checkpoint(self.rs, path, label)
        self.opponents[opp.id] = opp
        return opp

    def sprites(self) -> dict[str, str]:
        found = {}
        for d in reversed(self.cards_dirs):  # --cards wins over the bundled folder
            for f in d.iterdir():
                if f.suffix.lower() in SPRITE_EXT:
                    found[f.stem] = f"/cards/{f.name}"
        return found

    def sprite_file(self, name: str) -> Path | None:
        for d in self.cards_dirs:
            f = d / Path(name).name
            if f.is_file():
                return f
        return None

    def config(self) -> dict:
        rs = self.rs
        return {
            "cards": [{"id": i, "kind": KIND_NAMES[ct.kind], "color": ct.color, "rank": ct.rank,
                       "sprite": sprite_name(rs, i)} for i, ct in enumerate(rs.card_types)],
            "colors": list(COLOR_NAMES[: rs.colors]),
            "sprites": self.sprites(),
            "opponents": self.opponent_list(),
        }

    def opponent_list(self) -> list[dict]:
        return [{"id": o.id, "label": o.label, "kind": o.kind, "detail": o.detail} for o in self.opponents.values()]

    def new_game(self, opponent_id: str, seat) -> dict:
        opponent = self.opponents[opponent_id]
        human = random.randrange(2) if seat in (None, "random") else int(seat)
        while True:
            seed = random.randrange(2**31)
            try:
                game = Game(self.rs, opponent, human, seed)
                break
            except ValueError:  # a deal with no number card for the first discard
                continue
        gid = uuid.uuid4().hex
        self.games = {k: v for k, v in list(self.games.items())[-15:]}  # keep a few
        self.games[gid] = game
        start = game.view()  # before the bot's opening moves, so the page can deal and replay them
        return {"game": gid, "seat": human, "seed": seed, "opponent": opponent.label,
                "view": start, "steps": game.run_bot()}

    def act(self, gid: str, action: int) -> dict:
        game = self.games[gid]
        steps = game.act(action)
        return {"steps": steps, "view": game.view()}


class Handler(BaseHTTPRequestHandler):
    app: App

    def log_message(self, fmt, *args) -> None:  # quiet
        pass

    def _send(self, body: bytes, ctype: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status: int = 200) -> None:
        self._send(json.dumps(obj).encode(), "application/json", status)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send((STATIC / "index.html").read_bytes(), CONTENT_TYPES[".html"])
        elif self.path == "/api/config":
            with self.app.lock:
                self._json(self.app.config())
        elif self.path.startswith("/cards/"):
            f = self.app.sprite_file(self.path.removeprefix("/cards/"))
            if f is None:
                self._send(b"not found", "text/plain", 404)
            else:
                self._send(f.read_bytes(), CONTENT_TYPES.get(f.suffix.lower(), "application/octet-stream"))
        else:
            self._send(b"not found", "text/plain", 404)

    def do_POST(self) -> None:
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            with self.app.lock:
                match self.path:
                    case "/api/new":
                        self._json(self.app.new_game(body["opponent"], body.get("seat")))
                    case "/api/act":
                        self._json(self.app.act(body["game"], int(body["action"])))
                    case "/api/load":
                        opp = self.app.load(body["path"], body.get("label") or None)
                        self._json({"loaded": opp.id, "opponents": self.app.opponent_list()})
                    case _:
                        self._json({"error": "unknown endpoint"}, 404)
        except Exception as exc:  # a demo: report, don't crash the server
            self._json({"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.BAD_REQUEST)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="uno play", description="Play UNO against the rulebots and checkpoints.")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--ruleset", default="FULL_UNO", choices=list(NAMED))
    ap.add_argument("--ckpt", action="append", default=[], metavar="PATH[=LABEL]",
                    help="actor export (*_actor.pt) to offer as an opponent; repeatable")
    ap.add_argument("--cards", type=Path, default=None, help="directory of card sprites")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)

    app = App(NAMED[args.ruleset], args.cards)
    for spec in args.ckpt:
        path, _, label = spec.partition("=")
        opp = app.load(path, label or None)
        print(f"loaded {opp.label}: {opp.detail}")
    Handler.app = app
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"UNO table on {url}  ({args.ruleset}, {len(app.opponents)} opponents, {len(app.sprites())} sprites) -- Ctrl-C to stop")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, (url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
