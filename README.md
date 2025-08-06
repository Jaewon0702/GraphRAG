# What is GraphRAG and how it works?
Retrieval Augmented Generation (RAG) is a technique that connects external data sources to enhance the output of large language models (LLMs). This technique is perfect for LLMs to access private or domain-specific data and address hallucination issues. Therefore, RAG has been widely used to power many GenAI applications, such as AI chatbots and recommendation systems.       

A baseline RAG usually integrates a vector database and an LLM, where the vector database stores and retrieves contextual information for user queries, and the LLM generates answers based on the retrieved context. While this approach works well in many cases, it struggles with complex tasks like multi-hop reasoning or answering questions that require connecting disparate pieces of information.    

Unlike a baseline RAG that uses a vector database to retrieve semantically similar text, GraphRAG enhances RAG by incorporating knowledge graphs (KGs). Knowledge graphs are data structures that store and link related or unrelated data based on their relationships.    

A GraphRAG pipeline usually consists of two fundamental processes: indexing and querying.    

<img width="786" height="493" alt="image" src="https://github.com/user-attachments/assets/a216ec6c-b114-48c1-a9c7-f354e7b63e7d" />

The GraphRAG Pipeline (Image Source: [GraphRAG Paper](https://arxiv.org/pdf/2404.16130))  
