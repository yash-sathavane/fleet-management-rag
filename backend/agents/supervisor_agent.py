from information_agent import information_agent

def supervisor_agent(question):

    print("\nSupervisor Agent Received Query")

    answer = information_agent(question)

    print("Information Agent Completed")

    return answer