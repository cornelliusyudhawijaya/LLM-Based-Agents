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

gpt4_config = {
    "seed": 42,  # change the seed for different trials
    "temperature": 0,
    "config_list": config_list,
    "request_timeout": 120,
}

# engineer = autogen.AssistantAgent(
#     name="Engineer",
#     llm_config=gpt4_config,
#     system_message='''You are Machine Learning Engineer. You follow an approved plan. 
#     You write python/shell code to solve tasks, specifically to deploy the model provided by data scientist. 
#     Wrap the code in a code block that specifies the script type. 
#     The user can't modify your code. So do not suggest incomplete code which requires others to modify. 
#     Don't use a code block if it's not intended to be executed by the executor.
# Don't include multiple code blocks in one response. Do not ask others to copy and paste the result.
#  Check the execution result returned by the executor.
# If the result indicates there is an error, fix the error and output the code again.
#  Suggest the full code instead of partial code or code changes. 
#  If the error can't be fixed or if the task is not solved even after the code is executed successfully,
#  analyze the problem, revisit your assumption, collect additional info you need, 
#  and think of a different approach to try.
# ''',
# )

#Initiate the User Proxt Agent called Admin
user_proxy = autogen.UserProxyAgent(
   name="Admin",
   system_message="""A human admin. Interact with the planner to discuss the plan. Plan execution needs to be approved by this admin.
   Reply TERMINATE if the task has been solved at full satisfaction.
Otherwise, reply CONTINUE, or the reason why the task is not solved yet."
   """,
   code_execution_config={"work_dir": "data_sci"},
   human_input_mode="TERMINATE",
   
)

#Initiate the Data Scientist agent called Scientist
scientist = autogen.AssistantAgent(
    name="Scientist",
    llm_config=gpt4_config,
    system_message="""You are Data Scientist. You follow an approved plan. 
    you write python/shell code to solve tasks, specifically on data exploration, data analysis, data cleaning, feature engineering
    and modelling tasks. Make sure that the chosen model according to the performance.
    Wrap the code in a code block that specifies the script type. 
    The user can't modify your code. So do not suggest incomplete code which requires others to modify. 
    Don't use a code block if it's not intended to be executed by the executor.
    Don't include multiple code blocks in one response. Do not ask others to copy and paste the result.
    Check the execution result returned by the executor.
    If the result indicates there is an error, fix the error and output the code again.
    Suggest the full code instead of partial code or code changes. 
    If the error can't be fixed or if the task is not solved even after the code is executed successfully,
    analyze the problem, revisit your assumption, collect additional info you need, 
    and think of a different approach to try.
    """,
    code_execution_config={"work_dir": "data_sci"}
)

#Initiate the planning agent called Planer
planner = autogen.AssistantAgent(
    name="Planner",
    system_message='''Planner. Suggest a plan. 
    Revise the plan based on feedback from admin and critic, until admin approval.
The plan may involve scientist.
Explain the plan first. Be clear which step is performed by a scientist.
''',
    llm_config=gpt4_config,
)

#Initiate the agent that would execute the code called Executor
executor = autogen.UserProxyAgent(
    name="Executor",
    system_message="Executor. Execute the code written by scientist and save them in the file if it's execute fine. Report the result by breaking it down for each plan section.",
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "data_sci"},
)

#Initiate the agent that would provide feedback to the plan
critic = autogen.AssistantAgent(
    name="Critic",
    system_message="Critic. Double check plan, claims, code from other agents and provide feedback. Check whether the plan is viable or not.",
    llm_config=gpt4_config,
)
groupchat = autogen.GroupChat(agents=[user_proxy, scientist, planner, executor, critic], messages=[], max_round=100)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=gpt4_config)

user_proxy.initiate_chat(
    manager,
    message="""
Analyze this following data https://raw.githubusercontent.com/cornelliusyudhawijaya/Cross-Sell-Insurance-Business-Simulation/main/sample.csv 
and develop machine learning model to predict the response variable. 
""",
)