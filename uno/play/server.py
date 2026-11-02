"""`uv run uno play [--port 8765] [--ruleset FULL_UNO] [--ckpt PATH[=LABEL] ...] [--coach PATH[=LABEL] ...] [--cards DIR]`

Serves a one-page UNO table on localhost where you play against any rulebot or a
loaded actor export. Standard library only: the engine runs here, the page only
animates what the server reports.

Each request returns the *steps* since the human's last decision -- one per engine
action, with the public events it produced and the view after it -- so the page can
replay the bot's moves one by one. Opponent actions are sampled from the agent's
distribution; the demo is not an evaluation and uses no registered seeds.

Analysis (docs/play/plan.md): coaches give their distribution at every human
decision; reveal mode also sends the bot's hand and full policy; the value curve is
rollout values computed in a background thread (`uno.play.analysis`); the network
view traces one decision through the actor (`uno.play.inspect`).

Card sprites are optional: put files named like `red_7.png`, `blue_skip.png`,
`green_reverse.png`, `yellow_draw2.png`, `wild.png`, `wild_draw4.png` and `back.png`
(png/svg/webp/jpg) in `uno/play/static/cards/` or pass `--cards DIR`. Missing ones
are drawn in CSS.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
import random
import threading
import uuid
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from uno.game.engine import Phase, State, legal_actions, new_game, step
from uno.game.rules import NAMED, Kind, Play, Ruleset
from uno.play.analysis import rollout_values
from uno.play.opponents import NeuralPlayer, Opponent, Player, checkpoint, load_policy, rulebots

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


@dataclass
class Coach:
    """An analyser for the human's seat: its own `NeuralPlayer`, called once per human decision."""

    label: str
    level: str  # "L1" | "L2" (peeks at the bot's hand)
    actor: object  # ActorPolicy
    detail: str = ""

    def meta(self) -> dict:
        return {"label": self.label, "level": self.level, "detail": self.detail}


def _level(actor) -> str:
    return "L2" if actor.spec.has_opponent else "L1"


def _probs_json(probs: dict[int, float]) -> dict[str, float]:
    return {str(a): round(p, 5) for a, p in probs.items()}


class Game:
    """One game. `history` records every decision (both seats) with snapshots for analysis.

    Information boundaries (docs/play/plan.md section 2): each coach has its own player on
    the human seat, advanced exactly once per human decision whether or not the page shows
    it; the bot's inputs never change with `reveal`, which only changes what is *sent*;
    analysis runs on copies (`Player.copy`), never on the live players.
    """

    def __init__(self, rs: Ruleset, opponent: Opponent, human: int, seed: int,
                 coaches: list[Coach] = (), reveal: bool = False) -> None:
        self.rs, self.opponent, self.human, self.seed = rs, opponent, human, seed
        self.state: State = new_game(rs, seed)
        self.bot: Player = opponent.make(human ^ 1)
        self.rng = random.Random(seed ^ 0x5EED)
        self.reveal = reveal
        self.coaches = [(c, NeuralPlayer(c.actor, human)) for c in coaches]
        # The model that plays your seat in rollouts: the first L1 coach, else the bot's own policy.
        l1 = [(c, p) for c, p in self.coaches if c.level == "L1"]
        if l1:
            self.seat_label, self.seat_model, self.seat_extra = l1[0][0].label, l1[0][1], False
        else:
            self.seat_label, self.seat_model, self.seat_extra = opponent.label, opponent.make(human), True
        self.history: list[dict] = []
        self.values: dict[int, tuple[float, float]] = {}
        self.value_curve = False
        if self.state.player == human:
            self._record_human()

    def who(self, seat: int) -> str:
        return "you" if seat == self.human else "bot"

    # -- decisions --------------------------------------------------------------

    def _snapshot(self) -> list:
        players = [None, None]
        players[self.human] = self.seat_model.copy()
        players[self.human ^ 1] = self.bot.copy()
        return players

    def _record_human(self) -> None:
        s = self.state
        point = {"turn": s.turn, "who": "you", "state": s, "players": self._snapshot()}
        point["coach"] = {c.label: p.policy(s) for c, p in self.coaches}
        point["inputs"] = {c.label: p.last for c, p in self.coaches}
        if self.seat_extra:
            self.seat_model.policy(s)  # keep a stateful fallback's memory in step
        self.history.append(point)

    def _record_bot(self) -> dict:
        s = self.state
        point = {"turn": s.turn, "who": "bot", "state": s, "players": self._snapshot()}
        point["policy"] = self.bot.policy(s)
        point["inputs"] = {"bot": getattr(self.bot, "last", None)}
        self.history.append(point)
        return point

    def analysis(self) -> dict | None:
        """The pending human decision's coach distributions, or None."""
        s = self.state
        if s.terminal or s.player != self.human or not self.history:
            return None
        point = self.history[-1]
        return {"decision": len(self.history) - 1,
                "coach": {label: _probs_json(p) for label, p in point["coach"].items()}}

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
            "opp_hand": list(s.hands[me ^ 1]) if s.terminal or self.reveal else None,
            "analysis": self.analysis(),
        }

    def _apply(self, action: int, prob: float | None) -> dict:
        before = self.state
        seat = before.player
        decision = len(self.history) - 1
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
        self.history[decision]["action"] = action
        out = {"who": self.who(seat), "action": action, "prob": prob, "events": events, "decision": decision}
        if seat != self.human and self.reveal:
            point = self.history[decision]
            out["policy"] = _probs_json(point["policy"])
            out["opp_hand"] = list(before.hands[seat])  # the hand the bot chose from
        if not self.state.terminal and self.state.player == self.human:
            self._record_human()
        out["view"] = self.view()
        return out

    def act(self, action: int) -> list[dict]:
        if self.state.terminal or self.state.player != self.human:
            raise ValueError("not your turn")
        if action not in legal_actions(self.state):
            raise ValueError(f"illegal action {action}")
        return [self._apply(action, None)] + self.run_bot()

    def run_bot(self) -> list[dict]:
        steps = []
        while not self.state.terminal and self.state.player != self.human:
            probs = self._record_bot()["policy"]
            actions = list(probs)
            a = self.rng.choices(actions, weights=[probs[x] for x in actions])[0]
            steps.append(self._apply(a, probs[a]))
        return steps

    # -- values -------------------------------------------------------------------

    def value_points(self) -> list[dict]:
        return [{"decision": i, "turn": p["turn"], "who": p["who"], "value": v[0], "se": v[1]}
                for i, p in enumerate(self.history) if (v := self.values.get(i)) is not None]


class Rollouts(threading.Thread):
    """Background rollout values for games whose value curve is on, newest decision first."""

    def __init__(self, n: int) -> None:
        super().__init__(daemon=True)
        self.n = n
        self.cv = threading.Condition()
        self.jobs: list[tuple[Game, int]] = []
        self.current: Game | None = None

    def want(self, game: Game) -> None:
        with self.cv:
            self.current = game
            done = set(game.values)
            self.jobs = [(game, i) for i in range(len(game.history)) if i not in done]
            self.cv.notify()

    def run(self) -> None:
        import torch

        torch.set_num_threads(1)
        while True:
            with self.cv:
                while not self.jobs:
                    self.cv.wait()
                game, i = self.jobs.pop()  # newest first
            if i in game.values or game is not self.current or not game.value_curve:
                continue
            p = game.history[i]
            try:
                m, se = rollout_values([(p["state"], game.human, tuple(p["players"]))], self.n, game.seed * 7919 + i)
                game.values[i] = (float(m[0]), float(se[0]))
            except Exception as exc:  # keep serving; report once
                print(f"rollout failed at decision {i}: {type(exc).__name__}: {exc}")
                game.values[i] = (float("nan"), float("nan"))


class App:
    def __init__(self, rs: Ruleset, cards_dir: Path | None, rollouts: int = 96) -> None:
        self.rs = rs
        self.lock = threading.Lock()
        self.opponents: dict[str, Opponent] = {o.id: o for o in rulebots(rs)}
        self.coaches: list[Coach] = []
        self.games: dict[str, Game] = {}
        self.cards_dirs = [d for d in (cards_dir, STATIC / "cards") if d is not None and d.is_dir()]
        self.rollouts = Rollouts(rollouts)
        self.rollouts.start()
        from uno.play.inspect import Inspector

        self.inspector = Inspector(rs)

    def load(self, path: str, label: str | None = None) -> Opponent:
        opp = checkpoint(self.rs, path, label)
        self.opponents[opp.id] = opp
        self.prefit(opp.actor)
        return opp

    def prefit(self, actor) -> None:
        """Fit (or load) the network views' PCA and probes off the request path."""
        def run():
            try:
                self.inspector.bundle(actor)
            except Exception as exc:
                print(f"analysis fit failed: {type(exc).__name__}: {exc}")
        threading.Thread(target=run, daemon=True).start()

    def load_coach(self, path: str, label: str | None = None) -> Coach:
        actor, payload, detail = load_policy(self.rs, path)
        level = _level(actor)
        coach = Coach(label or f"{payload['name']} ({level})", level, actor, detail)
        self.coaches.append(coach)
        self.prefit(actor)
        return coach

    def game_coaches(self, opponent: Opponent) -> list[Coach]:
        """The loaded coaches; with no L1 coach, the checkpoint opponent coaches your seat."""
        coaches = list(self.coaches)
        if not any(c.level == "L1" for c in coaches) and opponent.actor is not None and _level(opponent.actor) == "L1":
            coaches.insert(0, Coach(f"{opponent.label}", "L1", opponent.actor, opponent.detail))
        return coaches

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
            "actions": [{"card": a.card, "color": a.color} if isinstance(a, Play) else
                        {"kind": "draw" if i == rs.draw_action else "keep"} for i, a in enumerate(rs.actions)],
            "sprites": self.sprites(),
            "opponents": self.opponent_list(),
            "coaches": [c.meta() for c in self.coaches],
            "rollouts": self.rollouts.n,
        }

    def opponent_list(self) -> list[dict]:
        return [{"id": o.id, "label": o.label, "kind": o.kind, "detail": o.detail} for o in self.opponents.values()]

    def new_game(self, opponent_id: str, seat, reveal: bool = False) -> dict:
        opponent = self.opponents[opponent_id]
        human = random.randrange(2) if seat in (None, "random") else int(seat)
        coaches = self.game_coaches(opponent)
        while True:
            seed = random.randrange(2**31)
            try:
                game = Game(self.rs, opponent, human, seed, coaches, reveal)
                break
            except ValueError:  # a deal with no number card for the first discard
                continue
        gid = uuid.uuid4().hex
        self.games = {k: v for k, v in list(self.games.items())[-15:]}  # keep a few
        self.games[gid] = game
        start = game.view()  # before the bot's opening moves, so the page can deal and replay them
        steps = game.run_bot()
        return {"game": gid, "seat": human, "seed": seed, "opponent": opponent.label, "reveal": reveal,
                "coaches": [c.meta() for c in coaches], "seat_model": game.seat_label,
                "view": start, "steps": steps}

    def act(self, gid: str, action: int) -> dict:
        game = self.games[gid]
        steps = game.act(action)
        if game.value_curve:
            self.rollouts.want(game)
        return {"steps": steps, "view": game.view()}

    def set_reveal(self, gid: str, reveal: bool) -> dict:
        game = self.games[gid]
        game.reveal = bool(reveal)
        return {"view": game.view()}

    def values(self, gid: str, on: bool | None) -> dict:
        game = self.games[gid]
        if on is not None:
            game.value_curve = bool(on)
        if game.value_curve:
            self.rollouts.want(game)
        return {"on": game.value_curve, "seat_model": game.seat_label, "n": self.rollouts.n,
                "points": game.value_points(), "decisions": len(game.history)}

    def inspect(self, gid: str, decision: int, who: str) -> dict:
        game = self.games[gid]
        point = game.history[decision]
        if point["who"] == "bot" and not (game.reveal or game.state.terminal):
            raise ValueError("the bot's network view is shown in reveal mode only")
        decision_inputs = point["inputs"].get(who)
        if decision_inputs is None:
            raise ValueError(f"no network inputs for {who!r} at decision {decision} (a rulebot?)")
        actor = game.bot.actor if who == "bot" else next(c.actor for c, _ in game.coaches if c.label == who)
        chosen = point.get("action") if who == "bot" else None
        # the decision-maker's opponent: the bot for a coach, you for the bot. Its true hand is
        # sent only when you may see it (reveal or game over; your own hand you always know).
        seat = game.human ^ 1 if who == "bot" else game.human
        truth = who == "bot" or game.reveal or game.state.terminal
        return self.inspector.view(actor, decision_inputs, chosen,
                                   {"state": point["state"], "seat": seat, "truth": truth})


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
                        self._json(self.app.new_game(body["opponent"], body.get("seat"), bool(body.get("reveal"))))
                    case "/api/reveal":
                        self._json(self.app.set_reveal(body["game"], body.get("reveal")))
                    case "/api/values":
                        self._json(self.app.values(body["game"], body.get("on")))
                    case "/api/inspect":
                        self._json(self.app.inspect(body["game"], int(body["decision"]), body["who"]))
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
    ap.add_argument("--coach", action="append", default=[], metavar="PATH[=LABEL]",
                    help="actor export to analyse your seat with; an arm-CRITIC export is the L2 coach that "
                         "peeks at the bot's hand. Repeatable. With no L1 coach, a checkpoint opponent coaches.")
    ap.add_argument("--rollouts", type=int, default=96, help="rollouts per value-curve point")
    ap.add_argument("--cards", type=Path, default=None, help="directory of card sprites")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)

    app = App(NAMED[args.ruleset], args.cards, args.rollouts)
    for spec in args.ckpt:
        path, _, label = spec.partition("=")
        opp = app.load(path, label or None)
        print(f"loaded {opp.label}: {opp.detail}")
    for spec in args.coach:
        path, _, label = spec.partition("=")
        coach = app.load_coach(path, label or None)
        print(f"coach {coach.label} [{coach.level}]: {coach.detail}")
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
