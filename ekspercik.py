import streamlit as st

from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOllama
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import CharacterTextSplitter

from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.embeddings import OpenAIEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="langchain_search_api_key_openai", type="password")
    ollama_model_name = st.text_input("Ollama Model Name", key="langchain_search_ollama_model_name", value="mistral:7b")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/TortillaZHawaii/ekspercik)"
    "[OG Repo](https://codespaces.new/streamlit/llm-examples)"

st.title("🔎 Ekspercik")

uploaded_file = st.file_uploader(
    "Wklej wykład",
    type=["pdf"],
    help="Wklej wykład, który chcesz przeszukać.",
    accept_multiple_files=False,
)

if not ollama_model_name and not openai_api_key:
    st.info("Prosze podaj nazwę modelu Ollama lub klucz OpenAPI w sidebarze, aby kontynuować")
    st.stop()

if ollama_model_name:
    persist_directory = f"./data/db_{ollama_model_name}"
    embeddings = OllamaEmbeddings(model=ollama_model_name)
else:
    persist_directory = f"./data/db_openai"
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-ada-002")

if "db" not in st.session_state:
    st.session_state["db"] = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

if uploaded_file:
    if st.session_state.get("uploaded_file", None) != uploaded_file:
        st.session_state["uploaded_file"] = uploaded_file

        with st.spinner("📥 Pobieram..."):
            tmp_location = f"/tmp/{uploaded_file.name}"
            with open(tmp_location, "wb") as f:
                f.write(uploaded_file.getbuffer())
        with st.spinner("👀 Czytam z PDFa..."):
            loader = UnstructuredPDFLoader(
                file_path=tmp_location, ocr_languages="eng+pl", strategy="ocr_only",
            )
            raw_documents = loader.load()
        with st.spinner("🔪 Dzielę na zdania..."):
            text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=120)
            documents = text_splitter.split_documents(raw_documents)
        with st.spinner("🔎 Zapisuje w bazie..."):
            db: Chroma = st.session_state["db"]
            db.add_documents(documents)
            db.persist()
            st.session_state["db"] = db
            st.success("🎉 Gotowe!")


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Heja, jestem chatbotem, który czyta PDFy. Jak mogę Ci pomóc?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Podsumuj wykład w jednym zdaniu"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if ollama_model_name:
        llm = ChatOllama(model=ollama_model_name)
    else:
        llm = ChatOpenAI(model_name="gpt-4", openai_api_key=openai_api_key, streaming=True)

    memory = ConversationBufferMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True,
        input_key='question', output_key='answer',
    )
    db = st.session_state["db"]

    retriever= db.as_retriever(
        search_type="similarity", k=5, return_metadata=True
    )
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        result = qa(st.session_state.messages, callbacks=[st_cb], return_only_outputs=True)

        response = result["answer"]
        sources = result["source_documents"]

        # write each source document in a new line
        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
        st.session_state.messages.append({"role": "assistant", "content": "📚 Oto źródła:"})
        for source in sources:
            st.session_state.messages.append({"role": "assistant", "content": source})
