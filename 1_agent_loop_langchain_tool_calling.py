from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL="qwen3:1.7b"

@tool
def get_product_price(product: str) -> float:
    """ Look up for a product price in the catalog"""
    print(f"  >>Executing get_product_price(product='{product})')")
    prices = {"laptop":1299.99, "headphones":359.95, "keyboard":89.90}
    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to the price and return the final price.
    Available discount tiers are: bronze, silver, gold."""
    print(f"  >>Executing apply_discount(price='{price}', discount_tier='{discount_tier}')')")
    discount_percentages = {
        "bronze": 5,
        "silver": 12,
        "gold": 23,
    }

    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount/100), 2)

    # --- Agent Loop ---

@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools=[get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}
    llm = init_chat_model(f"ollama:{MODEL}",temperature=0)
    # llm = init_chat_model(f"openai:gpt-5", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" *60)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful Shopping Assistance."
                "You have access to a product catalog tool"
                "and a discount tool.\n\n"
                "STRICT RULES- You must follow these exactly:\n"
                "1. Never guess or assume any product price."
                "You must call get_product_price first to get the real product price."
                "If the real product price don't found NOT proceed further & TELL  the user PRODUCT Doesn't exists. \n"
                "2. Only call apply_discount after you received a price from the get_product_price. Pass the "
                "exact price returned from the get_product_price - do NOT pass a made-up number.\n"
                "3. NEVER Calculate the discount by yourself using Math."
                "Always use the apply_discount tool.\n"
                "4. If the user doesn't specify any discount tier,"
                "ask them again which tier to use - do NOT assume one."
            )
        ),
        HumanMessage(content=question)
    ]
    for iteration in range(1,MAX_ITERATIONS+1):
        print(f"\n--- Iteration {iteration} ---")
        ai_message= llm_with_tools.invoke(messages)
        tool_calls= ai_message.tool_calls

      # If no tools calls, then finish the answer
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

    #     Process only the FIRST Tool call = force one tool call per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args",{})
        tool_call_id = tool_call.get("id")
        print(f" [Tool Selected]: {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found")

        observation = tool_to_use.invoke(tool_args)

        print(f"   [Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))

    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a pencil with silver  discount?")
