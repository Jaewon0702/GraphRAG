import os
import urllib.request
import asyncio
import pandas as pd
import tiktoken

from dotenv import load_dotenv
load_dotenv()

# Prepare input directory and download example text
index_root = os.getcwd()  

os.makedirs(os.path.join(index_root, 'input'), exist_ok=True)
url = "https://www.gutenberg.org/cache/epub/7785/pg7785.txt"
file_path = os.path.join(index_root, 'input', 'davinci.txt')
urllib.request.urlretrieve(url, file_path)

# Truncate the file to a reasonable length to save token cost
with open(file_path, 'r+', encoding='utf-8') as file:
    lines = file.readlines()
    file.seek(0)
    file.writelines(lines[:934])
    file.truncate()

# Imports from GraphRAG
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.input.loaders.dfs import store_entity_semantic_embeddings
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.question_gen.local_gen import LocalQuestionGen
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.vector_stores import LanceDBVectorStore

async def main():
    # Locate output directory
    output_dir = os.path.join(index_root, "output")
    subdirs = [os.path.join(output_dir, d) for d in os.listdir(output_dir)]
    latest_subdir = max(subdirs, key=os.path.getmtime)
    INPUT_DIR = os.path.join(latest_subdir, "artifacts")

    # Table names
    COMMUNITY_REPORT_TABLE = "create_final_community_reports"
    ENTITY_TABLE = "create_final_nodes"
    ENTITY_EMBEDDING_TABLE = "create_final_entities"
    RELATIONSHIP_TABLE = "create_final_relationships"
    TEXT_UNIT_TABLE = "create_final_text_units"
    COMMUNITY_LEVEL = 2

    # Load parquet files
    entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
    entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")
    entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)

    # Connect to LanceDB
    description_embedding_store = LanceDBVectorStore(
        uri="./lancedb",
        collection_name="entity_description_embeddings"
    )
    description_embedding_store.connect()

    # Embed entities
    entity_description_embeddings = store_entity_semantic_embeddings(
        entities=entities,
        vectorstore=description_embedding_store
    )

    # Load relationship data
    relationship_df = pd.read_parquet(f"{INPUT_DIR}/{RELATIONSHIP_TABLE}.parquet")
    relationships = read_indexer_relationships(relationship_df)

    # Load reports
    report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
    reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)

    # Load text units
    text_unit_df = pd.read_parquet(f"{INPUT_DIR}/{TEXT_UNIT_TABLE}.parquet")
    text_units = read_indexer_text_units(text_unit_df)

    # Setup OpenAI API
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set in environment variables.")

    llm = ChatOpenAI(
        api_key=api_key,
        model="gpt-3.5-turbo",
        api_type=OpenaiApiType.OpenAI,
        max_retries=20,
    )
    
    embedding_model = "text-embedding-3-small"
    token_encoder = tiktoken.get_encoding("cl100k_base")
    text_embedder = OpenAIEmbedding(
        api_key=api_key,
        api_base=None,
        api_type=OpenaiApiType.OpenAI,
        model=embedding_model,
        deployment_name=embedding_model,
        max_retries=20,
    )

    context_builder = LocalSearchMixedContext(
        community_reports=reports,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        covariates=None,
        entity_text_embeddings=description_embedding_store,
        embedding_vectorstore_key=EntityVectorStoreKey.ID,
        text_embedder=text_embedder,
        token_encoder=token_encoder,
    )

    local_context_params = {
        "text_unit_prop": 0.5,
        "community_prop": 0.1,
        "conversation_history_max_turns": 5,
        "conversation_history_user_turns_only": True,
        "top_k_mapped_entities": 10,
        "top_k_relationships": 10,
        "include_entity_rank": True,
        "include_relationship_weight": True,
        "include_community_rank": False,
        "return_candidate_context": False,
        "embedding_vectorstore_key": EntityVectorStoreKey.ID,
        "max_tokens": 12000,
    }

    llm_params = {
        "max_tokens": 2000,
        "temperature": 0.0,
    }

    # Perform search query
    search_engine = LocalSearch(
        llm=llm,
        context_builder=context_builder,
        token_encoder=token_encoder,
        llm_params=llm_params,
        context_builder_params=local_context_params,
        response_type="multiple paragraphs",
    )
    
    result = await search_engine.asearch("Tell me about Leonardo Da Vinci")
    print("Search Result:", result.response)

    # Generate follow-up questions
    question_generator = LocalQuestionGen(
        llm=llm,
        context_builder=context_builder,
        token_encoder=token_encoder,
        llm_params=llm_params,
        context_builder_params=local_context_params,
    )

    question_history = [
        "Tell me about Leonardo Da Vinci",
        "Leonardo's early works",
    ]

    candidate_questions = await question_generator.agenerate(
        question_history=question_history,
        context_data=None,
        question_count=5,
    )
    print("Generated Questions:")
    for q in candidate_questions.response:
        print("-", q)

# Entry point
if __name__ == "__main__":
    asyncio.run(main())