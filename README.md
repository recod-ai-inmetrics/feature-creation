### Utils files:
* ai_agent.py
* transform_codebase.py

#### 1. ai_agent.py
ai_agent.py is responsible for the agents creation. Here we create 2 principal classes:
* Chat: creates an agent that works like a normal chat in a Q&A interaction. Here's an example:
```python
role = 'Your agent role'
myChat = Chat(pipeline = yourHugginFacePipeline, useHistory = boolean, role = role)
// if you want the chat record to maintain the conversation, useHistory must be True. Otherwise, False.
question = 'Your question here'
answer = myChat.ask(query = question)
print(answer) // your answer
```
* CodeRAG: creates an agent that takes search queries and a question. Given the search queries, it will search for useful context and answer you based on the context. Here's an example:
```python
repo_path = 'link to a github repository that will be used to search for context. The repository will be cloned to a path specified by the user'
to_path = 'github repository destination path'
documentList = 'your langchain documents list' // the documents will be stored in the vector store
role = 'your agent role'
language = 'code language used in your code. If the documents are the code documentation, specify the language as text. The lib supports python and java too.'
embeddingModel = 'sentence-transformers/embeddingModelAvailableInSentenceTransformersLib'

ragAgent = CodeRAG(
    pipeline = yourHugginFacePipeline, 
    documentationList = documentList, 
    repo_path = repo_path,
    to_path = to_path,
    language = language,
    embeddingModel = embeddingModel,
    role = role
    )

question = 'your question'
searchQueries = 'list of your search queries'
```