Refector @backend/ai_generator.py to support recursive tool calling where based on the query, claude can make repeated tool calls. We will restrict the total tool call count to MAX_SEQUENTIAL_TOOL_CALL parameter in @backend/config.py (set default to 3). That means that Claude can either decide to answer the query after 0, 1, 2, or 3 tool calls. 

## Current behavior: 
The current behavior as implemented in @backend/ai_generator.py makes a maximum of 1 tool call and then remove tool call option after seeing the results and do not provide claude AI model an option to make 2nd tool call based on query complexity. 

## Desired behavior:
1. Each tool call should be a separate separate API request where claude can reason about previous tool call results, query, etc to decide if it needs to make additional tool call request. 
2. If claude model decides to not use a tool, then the system should return the generated response stored in response.content[0].text
3. This should help answer more complex user queries that compares different courses and provide detailed review a particular course. The model gets access to each of the results of previous tool invocation and summarizes the answer when it has full context. 
4. This is an agentic RAG behavior that I want to implement.

## Additions
1. Update the system prompt accordingly in @backend/ai_generator.py to support Agentic RAG pattern / behavior described above. 
2. Update the tests in @backend/test/test_ai_generator.py 
3. Write test that verifies external behavior (API calls made, tools executed, result returned) rather than internal state details. Use sample complex query such as -- what are key similarities and differences in the courses: (i) MCP: Build Rich-Context AI Apps with Anthropic, (ii) Advanced Retrieval for AI with Chroma -- this should atleast require 3 tool call invocation 



## Implmentation plan
1. Use two subagents to brainstorm different approaches to implement the desired behavior and let me select the correct option. Do not implement any code before I confirm. 


