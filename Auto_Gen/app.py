import autogen
api_key = ''
config_list = [
    {
        'model': 'gpt-4',
        'api_key': api_key,
    },  # OpenAI API endpoint for gpt-4
    {
        'model': 'gpt-3.5-turbo',
        'api_key': api_key,
    },  # OpenAI API endpoint for gpt-3.5-turbo
    {
        'model': 'gpt-3.5-turbo-16k',
        'api_key': api_key,
    },  # OpenAI API endpoint for gpt-3.5-turbo-16k
]

llm_config = {"config_list": config_list, "seed": 42}

# create an AssistantAgent instance named "assistant"
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config,
)
# create a UserProxyAgent instance named "user_proxy"
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={"work_dir": "webtask"},
    llm_config=llm_config,
    system_message="""Reply TERMINATE if the task has been solved at full satisfaction.
Otherwise, reply CONTINUE, or the reason why the task is not solved yet."""
)

# the assistant receives a message from the user, which contains the task description
user_proxy.initiate_chat(
    assistant,
    message="""
Can you analyze this data and give me your impression what data it is about: https://raw.githubusercontent.com/cornelliusyudhawijaya/Cross-Sell-Insurance-Business-Simulation/main/sample.csv.
""",
)