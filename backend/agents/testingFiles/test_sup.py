from supervisor_agent import supervisor_agent

question = input("Ask Question: ")

response = supervisor_agent(question)

print("\nFinal Answer:")
print(response)