from dataclasses import dataclass
from uno.rl.core.config import ModelConfig
from uno.rl.env.features import Spec
from uno.rl.model.relational import Actor as RelationalActor
from uno.rl.model.relational import Critic as RelationalCritic


def make_actor(sp: Spec, model_config: ModelConfig):
    match model_config.kind:
        case "relational":
            return RelationalActor(
                sp,
                width=model_config.width,
                blocks=model_config.blocks,
            )
        case _:
            raise ValueError(f"unknown model kind {model_config.kind!r}")


def make_critic(sp: Spec, model_config: ModelConfig):
    match model_config.kind:
        case "relational":
            return RelationalCritic(
                sp,
                width=model_config.width,
                blocks=model_config.blocks,
            )
        case _:
            raise ValueError(f"unknown model kind {model_config.kind!r}")


@dataclass(frozen=True)
class Shapes:
    """Recorded tensor shapes and parameter counts."""

    arm: str
    model_kind: str
    tokens: int
    colors: int
    slots: int
    wilds: int
    tok_width: int
    col_width: int
    glob_width: int
    actions: int
    actor_params: int
    critic_params: int

    def __str__(self) -> str:
        return (
            f"arm {self.arm} model {self.model_kind}: {self.tokens} tokens "
            f"({self.colors} x {self.slots} + {self.wilds}), channels {self.tok_width}/{self.col_width}/{self.glob_width}, "
            f"{self.actions} actions, actor {self.actor_params:,} params, critic {self.critic_params:,} params"
        )


def shapes(
    actor_spec: Spec, critic_spec: Spec, model_config: ModelConfig
) -> Shapes:
    lay = actor_spec.layout
    actor = make_actor(actor_spec, model_config)
    critic = make_critic(critic_spec, model_config)
    return Shapes(
        arm=actor_spec.arm, model_kind=model_config.kind, tokens=lay.n_tokens, colors=lay.colors, slots=lay.slots, wilds=lay.wilds,
        tok_width=actor_spec.tok_width, col_width=actor_spec.col_width, glob_width=actor_spec.glob_width, actions=lay.rs.num_actions,
        actor_params=sum(p.numel() for p in actor.parameters()), critic_params=sum(p.numel() for p in critic.parameters()),
    )
