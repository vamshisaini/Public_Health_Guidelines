
# app.py
# Public Health Guidelines Research Assistant

import os
import re
import pandas as pd
import streamlit as st

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

load_dotenv(override=False)
# Load Streamlit Cloud secrets into environment variables
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

if "LANGSMITH_API_KEY" in st.secrets:
    os.environ["LANGSMITH_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]

if "LANGSMITH_TRACING" in st.secrets:
    os.environ["LANGSMITH_TRACING"] = st.secrets["LANGSMITH_TRACING"]

os.environ["LANGSMITH_PROJECT"] = "Public-Health-RAG"

st.set_page_config(
    page_title="Public Health Research Assistant",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Public Health Guidelines Research Assistant")

st.write(
    "Ask questions about floods, heatwaves, infectious diseases, "
    "and public-health guidance using information retrieved from "
    "official WHO and Indian public-health webpages."
)

if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY is missing. Please check your .env file.")
    st.stop()

# --------------------------------------------------
# Gemini model selection
# --------------------------------------------------

st.sidebar.subheader("Gemini Model")

MODEL_OPTIONS = {
    "Gemini 3.6 Flash": "gemini-3.6-flash",
    "Gemini 3.7 Flash": "gemini-3.7-flash",
    "Gemini 3.8 Flash": "gemini-3.8-flash"
}

selected_model = st.sidebar.selectbox(
    "Choose Gemini model",
    list(MODEL_OPTIONS.keys()),
    index=2
)

model_name = MODEL_OPTIONS[selected_model]



# --------------------------------------------------
# 2. Load models and initialize Chroma database
# --------------------------------------------------

@st.cache_resource
def load_resources(model_name):

    # 1. Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    persist_directory = "./vectorstore/chroma_db"
    collection_name = "public_health_guidelines"

    # 2. Open existing Chroma database
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )

    # 3. Check whether documents are already indexed
    if vectorstore._collection.count() == 0:

        status = st.empty()
        status.info("Initializing the public-health knowledge base...")

        # Official source webpages
        sources = [
            {
                "url": "https://www.who.int/news-room/questions-and-answers/item/how-do-i-protect-my-health-in-a-flood/",
                "title": "How to protect your health in a flood",
                "organization": "WHO",
                "region": "Global",
                "topic": "Floods"
            },
            {
                "url": "https://www.who.int/europe/emergencies/situations/climate-crisis-extreme-weather/public-health-advice-on-protecting-your-health-before-during-and-after-floods",
                "title": "Public health advice before, during and after floods",
                "organization": "WHO Europe",
                "region": "European Region",
                "topic": "Floods"
            },
            {
                "url": "https://www.who.int/india/heat-waves",
                "title": "Heat Waves",
                "organization": "WHO India",
                "region": "India",
                "topic": "Heatwaves"
            },
            {
                "url": "https://www.who.int/en/news-room/fact-sheets/detail/climate-change-heat-and-health",
                "title": "Climate change: Heat and health",
                "organization": "WHO",
                "region": "Global",
                "topic": "Heat and Health"
            },
            {
                "url": "https://ncdc.mohfw.gov.in/includes/About/CentresAndDivision/leptospirosis.php",
                "title": "Leptospirosis",
                "organization": "NCDC India",
                "region": "India",
                "topic": "Infectious Diseases"
            }
        ]

        documents = []

        # 4. Load webpages and attach source metadata
        for source in sources:

            loader = WebBaseLoader(source["url"])
            loaded_docs = loader.load()

            for doc in loaded_docs:
                doc.metadata.update({
                    "title": source["title"],
                    "organization": source["organization"],
                    "region": source["region"],
                    "topic": source["topic"],
                    "url": source["url"]
                })

            documents.extend(loaded_docs)

        if not documents:
            raise RuntimeError(
                "No documents could be loaded from the source webpages."
            )

        # 5. Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=120
        )

        chunks = splitter.split_documents(documents)

        if not chunks:
            raise RuntimeError(
                "Document chunking produced no chunks."
            )

        # 6. Add documents and embeddings to Chroma
        vectorstore.add_documents(chunks)

        # 7. Verify database creation
        if vectorstore._collection.count() == 0:
            raise RuntimeError(
                "Chroma database initialization failed."
            )

        status.success(
            f"Knowledge base ready! {len(chunks)} chunks indexed."
        )

    return (
        vectorstore,
        ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.1
        ),
        CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    )



try:
    vectorstore, llm, reranker = load_resources(model_name)
except Exception as e:
    st.error(
        "Could not load the application resources. "
        "Check the database path, installed packages, API access, "
        "and model availability."
    )
    st.exception(e)
    st.stop()


# --------------------------------------------------
# 3. RAG prompts
# --------------------------------------------------

answer_prompt = ChatPromptTemplate.from_template("""
You are a Public Health Guidelines Research Assistant.

Answer the question using the provided context.

Rules:
- Use the context as the primary source.
- Do not invent facts or unsupported medical advice.
- If the context is insufficient, say so clearly.
- Distinguish India-specific guidance from global or
  European guidance.
- Give a clear, concise answer.
- Encourage professional medical assistance when appropriate.

Context:
{context}

Question:
{question}

Answer:
""")

answer_chain = answer_prompt | llm | StrOutputParser()

query_prompt = ChatPromptTemplate.from_template("""
Generate three alternative search queries for this public-health
question. Preserve the original meaning and use different wording.

Return only the queries, one per line.

Question: {question}
""")

query_chain = query_prompt | llm | StrOutputParser()


# --------------------------------------------------
# 4. Source formatting
# --------------------------------------------------

def format_context(documents):
    return "\n\n".join(
        f"Source: {doc.metadata.get('title', 'Unknown')}\n"
        f"Organization: {doc.metadata.get('organization', 'Unknown')}\n"
        f"Region: {doc.metadata.get('region', 'Not specified')}\n"
        f"Content: {doc.page_content}"
        for doc in documents
    )


def format_sources(documents, scores=None):
    sources = []
    seen_urls = set()

    for index, doc in enumerate(documents):
        url = doc.metadata.get("url", "")

        if url in seen_urls:
            continue

        seen_urls.add(url)

        source = {
            "Title": doc.metadata.get("title", "Unknown"),
            "Organization": doc.metadata.get("organization", "Unknown"),
            "Region": doc.metadata.get("region", "Not specified"),
            "Topic": doc.metadata.get("topic", "Not specified"),
            "URL": url
        }

        if scores is not None:
            source["Reranking score"] = round(
                float(scores[index]), 4
            )

        sources.append(source)

    return sources


# --------------------------------------------------
# 5. Basic RAG
# --------------------------------------------------

def basic_rag(question, k=3):
    documents = vectorstore.similarity_search(
        question,
        k=k
    )

    if not documents:
        return {
            "answer": "No relevant passages were found.",
            "sources": []
        }

    context = format_context(documents)

    answer = answer_chain.invoke({
        "context": context,
        "question": question
    })

    return {
        "answer": answer,
        "sources": format_sources(documents)
    }


# --------------------------------------------------
# 6. Advanced RAG
# --------------------------------------------------

def advanced_rag(question, k=3, top_n=3):
    # Query expansion
    expansion = query_chain.invoke({
        "question": question
    })

    alternative_queries = [
        line.strip().lstrip("0123456789.-) ")
        for line in expansion.splitlines()
        if line.strip()
    ][:3]

    queries = [question] + alternative_queries

    # Multi-query retrieval and deduplication
    candidates = []
    seen = set()

    for query in queries:
        documents = vectorstore.similarity_search(
            query,
            k=k
        )

        for doc in documents:
            key = (
                doc.metadata.get("url", ""),
                doc.page_content
            )

            if key not in seen:
                seen.add(key)
                candidates.append(doc)

    if not candidates:
        return {
            "answer": "No relevant passages were found.",
            "sources": []
        }

    # Reranking
    pairs = [
        (question, doc.page_content)
        for doc in candidates
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda item: item[1],
        reverse=True
    )[:top_n]

    ranked_documents = [item[0] for item in ranked]
    ranked_scores = [item[1] for item in ranked]

    context = format_context(ranked_documents)

    # Answer generation
    answer = answer_chain.invoke({
        "context": context,
        "question": question
    })

    return {
        "answer": answer,
        "sources": format_sources(
            ranked_documents,
            ranked_scores
        )
    }


# --------------------------------------------------
# 7. Full 20-question evaluation dataset
# --------------------------------------------------

evaluation_data = [
    ("How can people protect themselves during a flood?",
     "Floods",
     "Follow official flood warnings, avoid floodwater, and follow public-health and safety guidance."),

    ("What precautions should people take before a flood?",
     "Floods",
     "Prepare for flooding, follow official warnings, and plan how to stay safe."),

    ("What health risks can arise after a flood?",
     "Floods",
     "Floods can expose people to contaminated water, injuries, and other health hazards."),

    ("How should people approach drinking water after flooding?",
     "Floods",
     "Follow local water-safety advisories and use water confirmed safe for drinking."),

    ("What precautions should people take when cleaning a flooded home?",
     "Floods",
     "Follow official cleanup advice and avoid direct exposure to contaminated floodwater."),

    ("How can people protect themselves during a heatwave?",
     "Heatwaves",
     "Reduce heat exposure, stay hydrated when appropriate, and follow local heat-health advice."),

    ("What are common warning signs of heat-related illness?",
     "Heat and Health",
     "Recognize symptoms of heat-related illness and seek appropriate medical assistance when needed."),

    ("Who may be especially vulnerable during extreme heat?",
     "Heat and Health",
     "Older adults, young children, and people with certain health conditions may face increased risks."),

    ("How can people reduce their exposure to extreme heat?",
     "Heatwaves",
     "Limit exposure to high temperatures and follow local heatwave precautions."),

    ("How does extreme heat affect human health?",
     "Heat and Health",
     "Extreme heat can cause heat-related illness and worsen some existing health conditions."),

    ("What precautions should be considered during very hot weather in India?",
     "Heatwaves",
     "Follow local heat alerts and health advice, reduce heat exposure, and take appropriate precautions."),

    ("How can people reduce health risks during hot weather?",
     "Heat and Health",
     "Reduce heat exposure and follow evidence-based heat-health guidance."),

    ("What is leptospirosis?",
     "Infectious Diseases",
     "Leptospirosis is an infectious disease caused by Leptospira bacteria."),

    ("How can leptospirosis spread to humans?",
     "Infectious Diseases",
     "Infection can occur through contact with water or soil contaminated by infected animals' urine."),

    ("What symptoms may occur with leptospirosis?",
     "Infectious Diseases",
     "Symptoms can include fever, headache, and muscle pain; symptoms vary and require clinical assessment."),

    ("What precautions can reduce the risk of leptospirosis after flooding?",
     "Infectious Diseases",
     "Avoid potentially contaminated floodwater and follow public-health advice about exposure and prevention."),

    ("Why is contact with contaminated water a public-health concern?",
     "Floods",
     "Contaminated water may expose people to infectious agents and other hazards."),

    ("What should people do if they feel unwell after floodwater exposure?",
     "Infectious Diseases",
     "Seek medical advice and explain the exposure, particularly if symptoms develop."),

    ("How can communities reduce health risks associated with flooding?",
     "Floods",
     "Follow official warnings, maintain water safety, avoid flood hazards, and follow local public-health instructions."),

    ("Why is public-health guidance important during extreme weather?",
     "Heat and Health",
     "Public-health guidance helps people understand risks and take appropriate protective measures.")
]

evaluation_df = pd.DataFrame(
    evaluation_data,
    columns=["question", "expected_topic", "reference_answer"]
)


# --------------------------------------------------
# 8. Sidebar
# --------------------------------------------------

st.sidebar.header("Settings")

rag_version = st.sidebar.selectbox(
    "Choose RAG version",
    ["Basic RAG", "Advanced RAG"]
)

st.sidebar.caption(
    "Basic RAG uses direct vector retrieval. Advanced RAG "
    "adds query expansion and cross-encoder reranking."
)

mode = st.sidebar.radio(
    "Application mode",
    ["Ask a question", "Evaluate 20 questions"]
)

st.sidebar.divider()

st.sidebar.write("**Knowledge base**")
st.sidebar.write("5 official HTML webpages")
st.sidebar.write("Sentence Transformers + Chroma")
st.sidebar.write("Gemini for answer generation")

if os.getenv("LANGSMITH_API_KEY"):
    st.sidebar.success("LangSmith key detected")
else:
    st.sidebar.info("LangSmith tracing is optional in this app")


# --------------------------------------------------
# 9. Ask a question
# --------------------------------------------------

if mode == "Ask a question":
    question = st.text_area(
        "Enter your public-health question",
        placeholder="Example: How can people protect themselves during a heatwave?"
    )

    if st.button("Get Answer", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner(
                    f"Running {rag_version}..."
                ):
                    if rag_version == "Basic RAG":
                        result = basic_rag(question)
                    else:
                        result = advanced_rag(question)

                st.subheader("Answer")
                st.write(result["answer"])

                st.subheader("Retrieved sources")

                if result["sources"]:
                    st.dataframe(
                        pd.DataFrame(result["sources"]),
                        use_container_width=True,
                        hide_index=True
                    )

                    for source in result["sources"]:
                        if source["URL"]:
                            st.markdown(
                                f"- [{source['Title']}]({source['URL']})"
                            )
                else:
                    st.info("No sources were retrieved.")

            except Exception as e:
                error_message = str(e)

                if any(term in error_message.lower() for term in [
                    "resource_exhausted",
                    "quota",
                    "rate limit",
                    "429"
                ]):
                    st.error(
                        "The language model quota appears to be "
                        "exhausted. Please retry when quota is available."
                    )
                else:
                    st.error("The request failed.")
                    st.exception(e)


# --------------------------------------------------
# 10. Evaluate the 20 questions
# --------------------------------------------------

else:
    st.subheader("RAG Evaluation")
    st.write(
        "Run the 20-question evaluation set using the selected RAG "
        "version. Each question makes a live model request. "
        "Advanced RAG also makes query-expansion calls."
    )

    evaluation_count = st.selectbox(
        "Number of questions to run",
        [3, 5, 10, 20],
        index=0
    )

    st.caption(
        "Start with 3 questions if your Gemini quota is limited. "
        "Choose 20 when sufficient quota is available."
    )

    if st.button("Run Evaluation", type="primary"):
        evaluation_rows = []

        progress = st.progress(0)
        status = st.empty()

        for index, row in evaluation_df.head(
            evaluation_count
        ).iterrows():

            question = row["question"]
            status.write(
                f"Evaluating question "
                f"{len(evaluation_rows) + 1}/{evaluation_count}"
            )

            try:
                if rag_version == "Basic RAG":
                    result = basic_rag(question)
                else:
                    result = advanced_rag(question)

                answer = result["answer"]
                sources = result["sources"]
                error = ""

            except Exception as e:
                answer = ""
                sources = []
                error = str(e)[:300]

            evaluation_rows.append({
                "question": question,
                "expected_topic": row["expected_topic"],
                "reference_answer": row["reference_answer"],
                "generated_answer": answer,
                "sources": "; ".join(
                    source["Title"] for source in sources
                ),
                "source_urls": "; ".join(
                    source["URL"] for source in sources
                    if source["URL"]
                ),
                "error": error
            })

            progress.progress(
                len(evaluation_rows) / evaluation_count
            )

            # Stop early after a likely quota error
            if error and any(term in error.lower() for term in [
                "resource_exhausted",
                "quota",
                "rate limit",
                "429"
            ]):
                status.error(
                    "Quota or rate limit reached. "
                    "Evaluation stopped to avoid repeated failures."
                )
                break

        status.empty()

        results_df = pd.DataFrame(evaluation_rows)

        if not results_df.empty:
            successful = results_df[
                results_df["generated_answer"].str.len() > 0
            ]

            st.subheader("Evaluation results")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Questions attempted",
                len(results_df)
            )

            col2.metric(
                "Successful answers",
                len(successful)
            )

            col3.metric(
                "Failed answers",
                len(results_df) - len(successful)
            )

            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True
            )

            csv_data = results_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "Download evaluation results (CSV)",
                data=csv_data,
                file_name="public_health_evaluation_results.csv",
                mime="text/csv"
            )

            st.info(
                "These results record generated answers and their "
                "sources. They do not, by themselves, establish "
                "medical correctness. Review answer quality and "
                "retrieval relevance before reporting final metrics."
            )
