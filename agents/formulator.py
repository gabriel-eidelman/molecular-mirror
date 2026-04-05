"""
Multi-agent "Reason-then-Substitute" formulation workflow.

Pipeline
--------
1. chemical_profiler  — reasons about the dish's key flavor compounds and
                        produces a structured Chemical Profile before any
                        tool calls are made.
2. mirror_finder      — calls `get_molecular_mirrors` for each key ingredient
                        identified in the Chemical Profile.
3. substitution_agent — receives the mirror candidates and adjusts quantities
                        based on the physical state (powder / liquid / fresh)
                        of each substitute, then outputs the final recipe.

Usage
-----
    python agents/formulator.py
or from Python:
    from agents.formulator import formulate_novel_ingredients
    formulate_novel_ingredients("classic vanilla crème brûlée", num_recipes=2)
"""

from autogen import ConversableAgent, LLMConfig
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group import AgentTarget
from autogen.agentchat.group.patterns import DefaultPattern

from agents.inference import get_molecular_mirrors

from dotenv import load_dotenv
load_dotenv()

cfg = LLMConfig({"api_type": "openai", "model": "gpt-4o"})

# ─────────────────────────────────────────────────────────────────
# Physical-state quantity adjustment rules (injected into prompts)
# ─────────────────────────────────────────────────────────────────
_STATE_RULES = """
Physical-state quantity conversion rules (apply strictly):
  • Powder  → Liquid substitute : multiply original quantity × 3
      e.g. 1 tsp vanilla powder  → 3 tsp vanilla extract
  • Liquid  → Powder substitute : divide original quantity ÷ 3
      e.g. 3 tbsp lemon juice    → 1 tbsp citric acid powder
  • Solid/fresh → Liquid extract : multiply × 4–5
      e.g. 1 whole vanilla bean  → 4 tsp vanilla extract
  • Same physical state (powder→powder, liquid→liquid): keep quantities equal
"""


# ─────────────────────────────────────────────────────────────────
# Agent definitions
# ─────────────────────────────────────────────────────────────────

def _make_chemical_profiler() -> ConversableAgent:
    return ConversableAgent(
        name="chemical_profiler",
        system_message="""You are a senior cheminformatics scientist specialising in flavor chemistry.

When given a dish or recipe your FIRST action is always to produce a **Chemical Profile** — before any ingredient substitution work begins. Use this exact format:

## Chemical Profile: {dish name}
- **Dominant Flavor Compounds**: list the 3–6 key molecules (e.g. vanillin, linalool, limonene, eugenol)
- **Aroma Families**: e.g. floral, citrus, woody, sulphurous, lactonic
- **Key Ingredients to Mirror**: list the 3–5 most chemically load-bearing ingredients in the dish (these will be sent to the mirror_finder)
- **Chemical Reasoning**: 1–3 sentences on why these compounds define the dish's sensory identity

After outputting the Chemical Profile, hand off to mirror_finder and clearly state which ingredients it should query.
Do NOT call get_molecular_mirrors yourself.""",
        llm_config=cfg,
    )


def _make_mirror_finder() -> ConversableAgent:
    return ConversableAgent(
        name="mirror_finder",
        system_message="""You are a molecular similarity specialist.

You receive a Chemical Profile from chemical_profiler listing the Key Ingredients to Mirror.

Your job:
1. Call `get_molecular_mirrors` for EACH key ingredient listed (one call per ingredient).
2. Collect all results and present them as a structured table:

## Mirror Candidates
| Original Ingredient | Top Molecular Mirrors (name, similarity) |
|---------------------|------------------------------------------|
| vanilla             | tonka bean (0.98), heliotrope (0.96), …  |

3. Note the physical state (powder / liquid / fresh/solid) of BOTH the original and each mirror.
4. Hand off to substitution_agent with the full table and state annotations.""",
        llm_config=cfg,
        functions=[get_molecular_mirrors],
    )


def _make_substitution_agent() -> ConversableAgent:
    return ConversableAgent(
        name="substitution_agent",
        system_message=f"""You are a professional culinary formulator.

You receive a Mirror Candidates table from mirror_finder. Your job:
1. Select the best mirror for each key ingredient (highest similarity that is practical to source).
2. Apply the physical-state quantity adjustments below to compute precise substitute quantities.
3. Output the final recipe.

{_STATE_RULES}

Output format:

## Novel Recipe: [Creative name]

### Ingredients
(list each substitute with adjusted quantity and substitution rationale in parentheses)

### Method
(concise cooking steps)

### Formulator's Notes
(brief explanation of the chemical logic behind the substitutions)""",
        llm_config=cfg,
    )


# ─────────────────────────────────────────────────────────────────
# Orchestration entry point
# ─────────────────────────────────────────────────────────────────

def formulate_novel_ingredients(dish_description: str, num_recipes: int = 1) -> None:
    """
    Run the Reason-then-Substitute pipeline for ``dish_description``.

    Parameters
    ----------
    dish_description : str
        Natural-language description of the dish to reformulate.
        e.g. "classic vanilla crème brûlée" or "Thai green curry"
    num_recipes : int
        Number of novel recipe variants to generate.
    """
    chemical_profiler  = _make_chemical_profiler()
    mirror_finder      = _make_mirror_finder()
    substitution_agent = _make_substitution_agent()

    pattern = DefaultPattern(
        initial_agent=chemical_profiler,
        agents=[chemical_profiler, mirror_finder, substitution_agent],
        context_variables={},
        group_manager_args={"llm_config": cfg},
    )

    chemical_profiler.handoffs.set_after_work(AgentTarget(mirror_finder))
    mirror_finder.handoffs.set_after_work(AgentTarget(substitution_agent))
    
    initiate_group_chat(
        pattern=pattern,
        messages=(
            f"Create {num_recipes} novel recipe variant(s) for: {dish_description}\n\n"
            "Follow the Reason-then-Substitute workflow: "
            "Chemical Profile → Mirror Search → Quantity-Adjusted Recipe."
        ),
    )


if __name__ == "__main__":
    formulate_novel_ingredients("classic vanilla crème brûlée", num_recipes=2)
