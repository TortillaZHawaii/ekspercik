import streamlit as st

from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOllama
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OllamaEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="langchain_search_api_key_openai", type="password", disabled=True)
    ollama_model_name = st.text_input("Ollama Model Name", key="langchain_search_ollama_model_name", value="mistral:7b")
    # "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    # "[View the source code](https://github.com/streamlit/llm-examples/blob/main/pages/2_Chat_with_search.py)"
    # "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("🔎 Ekspercik")

uploaded_file = st.file_uploader(
    "Wklej wykład",
    type=["pdf"],
    help="Wklej wykład, który chcesz przeszukać.",
    accept_multiple_files=False,
)

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
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            documents = text_splitter.split_documents(raw_documents)
        with st.spinner("🔎 Zapisuje w bazie..."):
            embeddings = OllamaEmbeddings(model=ollama_model_name)
            db = Chroma.from_documents(documents, embeddings)
            st.session_state["db"] = db
            st.success("🎉 Gotowe!")


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search PDF. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Who won the Women's U.S. Open in 2018?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not ollama_model_name:
        st.info("Please specify Ollama model to continue to continue.")
        st.stop()

    if ollama_model_name:
        llm = ChatOllama(model=ollama_model_name)
    else:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)

    if st.session_state.get("uploaded_file", None) is None:
        st.info("Please upload a PDF to continue.")
        st.stop()



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

        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

        st.write(response)
        st.write("Źródła:")
        st.write(sources)
