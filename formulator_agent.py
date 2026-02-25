# Multi-agent orchestration for finding novel formulations using inference

from autogen import ConversableAgent, LLMConfig
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group import FunctionTarget
from autogen.agentchat.group.patterns import DefaultPattern
from inference import get_molecular_mirrors

from dotenv import load_dotenv
load_dotenv()

cfg = LLMConfig({"api_type": "openai", "model": "gpt-5"})

def formulate_novel_ingredients(num_rounds: int):
    novel_ingredients_agent = ConversableAgent(
        name="novel_ingredients_agent",
        system_message="""
        You have access to the function `get_molecular_mirrors`. This function accepts the name of a food item as input,
        and outputs a list of molecularly similar food items. 

        Your goal is to create novel alternative recipes for mimicking the taste of food items based on molecular similarities.

        You can run calls to get_molecular_mirrors several times until you are satisfied with the recipe you create.
        Output the new recipe.
        """,
        llm_config=cfg,
        functions=[get_molecular_mirrors]
    )

    pattern = DefaultPattern(
        initial_agent=novel_ingredients_agent,
        agents=[novel_ingredients_agent],
        context_variables={},
        group_manager_args={"llm_config": cfg}
    )
    initiate_group_chat(
        pattern=pattern,
        messages=f"""Create {num_rounds} new recipes.
        """
    )

if __name__ == "__main__":
    formulate_novel_ingredients(4)