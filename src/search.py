import os
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()


PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

AIMODEL = os.getenv("OPENAI_EMBEDDING_MODEL")

def search_prompt(question) -> str: 
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    embeddings = OpenAIEmbeddings(model=AIMODEL)

    store = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
        )
    

    results = store.similarity_search_with_score(question, k=10)
    contexto_docs = []
    for doc, score in results:
      contexto_docs.append(doc.page_content.strip())

    contexto = "\n\n".join(contexto_docs)

    formatted_prompt = prompt_template.format_prompt(
         contexto=contexto,
         pergunta=question)
    

    print("\nProcessando sua pergunta...")
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
    response = llm.invoke(formatted_prompt.to_messages())

    return response.content